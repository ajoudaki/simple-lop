import numpy as np
import math
import argparse
import sys
import time
import os

# Import PyTorch libraries
try:
    import torch
    import torch.nn as nn
    import torch.nn.init as init
    import torch.func as func
except ImportError:
    print("Error: PyTorch is required to run this script. Please install it.")
    # Exit if running as the main script
    if __name__ == "__main__":
        sys.exit(1)
    # If imported, execution will fail later if torch is used.
    pass

# --- SCRIPT CONFIGURATION ---
# Set the default device. Prioritize CUDA if available.
try:
    if torch.cuda.is_available():
        DEFAULT_DEVICE = "cuda:0"
    else:
        DEFAULT_DEVICE = "cpu"
except NameError:
    DEFAULT_DEVICE = "cpu"


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# START: Utilities and Model Definitions
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def _pytree_utils(pytree, dtype):
    """
    Manually flattens a 'pytree' (e.g., dict or tuple of tensors) into a
    single 1D vector and returns a function to un-flatten it.
    
    Ensures consistent ordering.
    """
    original_keys = []
    original_shapes = []
    num_elements_per_tensor = []
    
    # Handle different Pytree structures (dicts or tuples/lists)
    if isinstance(pytree, dict):
        # Sort dictionary items for consistency
        items = sorted(pytree.items())
        keys = [k for k, v in items]
        tensors = [v for k, v in items]
    else:
        # Assume it's an iterable of tensors (like a tuple from func.grad)
        tensors = list(pytree)
        keys = range(len(tensors))
        items = zip(keys, tensors)

    for key, tensor in items:
        original_keys.append(key)
        original_shapes.append(tensor.shape)
        num_elements_per_tensor.append(tensor.numel())

    # Handle empty pytree
    if not original_keys:
        # Try to infer device if possible, otherwise default to cpu
        device = 'cpu'
        if tensors:
             first_tensor = tensors[0]
             if isinstance(first_tensor, torch.Tensor):
                 device = first_tensor.device
        
        flat_tensor = torch.empty(0, device=device, dtype=dtype)
    else:
        flat_tensor = torch.cat([
            tensor.reshape(-1) for tensor in tensors
        ])

    def unflatten_fn(flat_tensor_in):
        if flat_tensor_in.numel() == 0 and not original_keys:
            return {} if isinstance(pytree, dict) else ()
            
        chunks = torch.split(flat_tensor_in, num_elements_per_tensor)
        
        if isinstance(pytree, dict):
            return {
                key: chunk.reshape(shape)
                for key, chunk, shape in zip(original_keys, chunks, original_shapes)
            }
        else:
            # Return a tuple if the original structure was iterable
            return tuple(chunk.reshape(shape) for chunk, shape in zip(chunks, original_shapes))

    return flat_tensor, unflatten_fn

class MLP(nn.Module):
    """PyTorch MLP model"""
    def __init__(self, d=2, h=100, L=5, activation='tanh'):
        super().__init__()
        self.L = L # Number of hidden layers
        self.h = h # Hidden dimension

        # Set activation function
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'gelu': nn.GELU()
        }
        if activation.lower() not in activations:
            raise ValueError(f"Unknown activation: {activation}. Choose from {list(activations.keys())}")
        self.activation = activations[activation.lower()]

        self.layers = nn.ModuleList()
        # Input layer
        self.layers.append(nn.Linear(d, h))
        # Hidden layers
        for _ in range(L - 1):
            self.layers.append(nn.Linear(h, h))
        # Output layer
        self.output_layer = nn.Linear(h, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Apply Xavier (Glorot) normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, X, capture=False):
        """
        Forward pass.
        """
        A = X
        for layer in self.layers:
            A = self.activation(layer(A))
        logits = self.output_layer(A)
        return None, logits # Return (p, logits) structure for compatibility

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# END: Utilities and Model Definitions
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# START: Cubic Subproblem Solver Implementation
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class CubicSubproblemSolver:
    """
    Encapsulates methods for solving the Cubic Regularized Newton subproblem
    using only Hessian-vector products (Hvps).
    
    Problem: min_p g^T p + 0.5 p^T H p + M/6 ||p||^3
    """

    def __init__(self, model, loss_fn, device='cpu'):
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        # Ensure consistent ordering for flattening/unflattening
        self.params_dict = dict(sorted(self.model.named_parameters()))
        self.buffers_dict = dict(sorted(self.model.named_buffers()))
        # Infer dtype
        try:
            self.dtype = next(self.model.parameters()).dtype
        except StopIteration:
            self.dtype = torch.float32 # Default if model has no parameters

    def _get_grad_and_hvp_fn(self, x_batch, y_batch):
        """
        Returns the flat gradient vector (g) and the HVP function using torch.func.
        """

        def stateless_loss_fn(params, buffers, x, y):
            # Ensure params/buffers are correctly referenced if they happen to be passed as empty
            if not params: params = self.params_dict
            if not buffers: buffers = self.buffers_dict
                
            # Functional call executes the model forward pass
            _, logits = func.functional_call(self.model, (params, buffers), (x,))
            return self.loss_fn(logits.squeeze(), y.squeeze())

        # Get the flattening utilities based on the current parameters
        flat_params, unflatten_fn = _pytree_utils(self.params_dict, dtype=self.dtype)
        dim = flat_params.numel()

        # 1. Calculate Gradient
        grad_fn = func.grad(stateless_loss_fn, argnums=0)
        primals_in = (self.params_dict, self.buffers_dict, x_batch, y_batch)
        
        # Compute and flatten the gradient
        grad_pytree = grad_fn(*primals_in)
        flat_grad, _ = _pytree_utils(grad_pytree, dtype=self.dtype)

        # 2. Define HVP function (using JVP of the gradient function)
        def hvp_fn_flat(v_flat):
            v_pytree = unflatten_fn(v_flat)
            
            # Define tangents corresponding to primals_in
            # Tangents for non-differentiated inputs (buffers, x, y) must be zero-like.
            tangents = (
                v_pytree, 
                {}, # Assuming buffers do not require gradients and can be represented by empty dict
                torch.zeros_like(x_batch), 
                torch.zeros_like(y_batch)
            )
            
            # JVP(grad(f)) = HVP
            # This standard approach is generally robust in modern PyTorch.
            _, hvp_pytree = func.jvp(grad_fn, primals_in, tangents)
            
            hvp_flat, _ = _pytree_utils(hvp_pytree, dtype=self.dtype)
            return hvp_flat

        return flat_grad, hvp_fn_flat, dim

    # ========================================================================
    # Utility: Lanczos Algorithm
    # ========================================================================

    def _run_lanczos(self, hvp_fn, dim, k_max, v_start=None, reorthogonalize=False):
        """
        Runs the Lanczos algorithm.
        
        Args:
            reorthogonalize: If True, performs full re-orthogonalization.
        
        Returns:
            If reorthogonalize=True, returns Q (list of vectors) and T (matrix).
            If reorthogonalize=False, returns T (matrix) only (memory efficient).
        """
        Q = []
        alphas = []
        betas = []

        # Initialize starting vector
        if v_start is None:
            v = torch.randn(dim, device=self.device, dtype=self.dtype)
        else:
            v = v_start.clone()
            
        v = v / torch.norm(v)
        
        # If we need to reorthogonalize, we must store the basis vectors.
        if reorthogonalize:
            Q.append(v)

        v_prev = torch.zeros_like(v)

        for j in range(k_max):
            # HVP calculation should not track gradients
            with torch.no_grad():
                w = hvp_fn(v)
                alpha = torch.dot(w, v)
                alphas.append(alpha.item())

                # Standard three-term recurrence
                w = w - alpha * v 
                if j > 0:
                    w = w - betas[j-1] * v_prev

                # Full Re-orthogonalization (Crucial for stability)
                if reorthogonalize:
                    # Double pass (Iterated Classical Gram-Schmidt) for numerical robustness
                    for _ in range(2):
                        for q_i in Q:
                            # w = w - (w^T q_i) q_i
                            w = w - torch.dot(w, q_i) * q_i

                beta = torch.norm(w)

            # Check for invariant subspace (breakdown)
            if beta < 1e-10:
                break

            if j < k_max - 1:
                betas.append(beta.item())
                v_prev = v
                v = w / beta
                if reorthogonalize:
                    Q.append(v)

        # Form the tridiagonal matrix T
        k = len(alphas)
        if k == 0:
            T = torch.zeros((0, 0), device=self.device, dtype=self.dtype)
        else:
            T = torch.diag(torch.tensor(alphas, device=self.device, dtype=self.dtype))
            if k > 1:
                betas_t = torch.tensor(betas, device=self.device, dtype=self.dtype)
                T += torch.diag(betas_t, diagonal=1)
                T += torch.diag(betas_t, diagonal=-1)

        if reorthogonalize:
            return Q, T
        else:
            return T

    # ========================================================================
    # Utility: Conjugate Gradient (CG)
    # ========================================================================

    def _cg_solve(self, A_fn, b, x0=None, tol=1e-6, max_iters=None):
        """
        Conjugate Gradient (CG) solver for Ax=b. Assumes A is Symmetric Positive Definite (SPD).
        """
        if max_iters is None:
            max_iters = b.numel()

        # Initialization
        if x0 is None:
            x = torch.zeros_like(b)
            r = b.clone()
            hvp_calls = 0
        else:
            # Warm start: Initialize x and recalculate the true residual
            x = x0.clone()
            r = b - A_fn(x)
            hvp_calls = 1

        p = r.clone()
        rr = torch.dot(r, r)
        b_norm = torch.norm(b)
        
        if b_norm == 0:
            return torch.zeros_like(b), hvp_calls

        # Relative tolerance stopping criterion
        stop_tol = tol * b_norm

        for i in range(max_iters):
            if torch.sqrt(rr) < stop_tol:
                break

            Ap = A_fn(p)
            hvp_calls += 1
            
            pAp = torch.dot(p, Ap)

            # Check for non-positive curvature (A must be PD for CG)
            # Robust check using a tolerance scaled by the norm of p
            if pAp <= 1e-12 * torch.norm(p)**2:
                # print(f"Warning: Non-positive curvature ({pAp:.2e}) detected in CG.")
                if i == 0 and x0 is None:
                    # If first iteration and no warm start, return the steepest descent direction as fallback
                    return b, hvp_calls 
                else:
                    # Otherwise, return the best estimate so far
                    return x, hvp_calls

            # CG Updates
            alpha = rr / pAp
            x = x + alpha * p
            r = r - alpha * Ap

            rr_new = torch.dot(r, r)
            beta = rr_new / rr
            p = r + beta * p
            rr = rr_new

        return x, hvp_calls

    # ========================================================================
    # Strategy 1: Direct Lanczos Projection
    # ========================================================================

    def _solve_tridiagonal_subproblem(self, T, g_norm, M, tol=1e-6, max_iters=50):
        """
        Solves the small projected cubic subproblem exactly using Newton's method
        on the secular equation: Psi(lambda) = ||y(lambda)|| - 2*lambda/M = 0
        """
        k = T.shape[0]
        # Eigendecomposition of T (T = U D U^T).
        D, U = torch.linalg.eigh(T)

        # Calculate z_tilde = U^T (||g|| e1).
        # By convention (torch.linalg.eigh), U has eigenvectors in columns.
        # U^T e1 corresponds to the first row of U.
        z_tilde = U[0, :] * g_norm
        z_tilde_sq = z_tilde**2

        lambda_min = D[0].item()

        # Initialize lambda. Must be >= -lambda_min.
        # We add a small safeguard to ensure strict positivity of (D + lambda I).
        safeguard = 1e-8
        lambda_boundary = max(0.0, -lambda_min)
        lambda_k = lambda_boundary + safeguard

        # Newton iteration for the root of Psi(lambda)
        for i in range(max_iters):
            D_lambda = D + lambda_k

            # Handling the "hard case" (lambda_k close to -lambda_min).
            # If too close to the pole, numerical instability arises. Perturb lambda slightly.
            if torch.any(D_lambda <= safeguard):
                 # print("Warning: Near the hard case. Perturbing lambda.")
                 lambda_k = lambda_boundary + safeguard
                 D_lambda = D + lambda_k

            inv_D_lambda = 1.0 / D_lambda

            # ||y(lambda)||^2 = sum_j z_tilde_j^2 / (d_j + lambda)^2
            y_norm_sq = torch.dot(z_tilde_sq, inv_D_lambda**2)
            y_norm = torch.sqrt(y_norm_sq)

            Psi = y_norm - (2 * lambda_k / M)

            # Convergence check (using relative tolerance)
            if abs(Psi) < tol * (1 + y_norm):
                break

            # Derivative Psi'(lambda) = - (1/||y||) * sum_j z_tilde_j^2 / (d_j + lambda)^3 - 2/M
            d_Psi = -torch.dot(z_tilde_sq, inv_D_lambda**3) / y_norm - (2 / M)

            # Newton step
            lambda_k_new = lambda_k - Psi / d_Psi

            # Safeguard lambda (ensure PSD constraint is maintained)
            lambda_k = max(lambda_k_new, lambda_boundary + safeguard)

        # Compute the optimal y* = -U (D+lambda I)^-1 z_tilde
        D_lambda = D + lambda_k
        # Final safeguard before inversion
        D_lambda[D_lambda <= safeguard] = safeguard
        
        # Optimized calculation: y* = -U @ diag(1/D_lambda) @ z_tilde
        y_star = -U @ ((1.0 / D_lambda) * z_tilde)

        return y_star, lambda_k

    def solve_direct_lanczos(self, x_batch, y_batch, M, k_max):
        """
        Strategy 1: Direct Lanczos Projection. O(Nk) memory.
        """
        print(f"\nRunning Strategy 1: Direct Lanczos Projection (k_max={k_max}, M={M})")
        start_time = time.time()
        hvp_count = 0

        g, hvp_fn, dim = self._get_grad_and_hvp_fn(x_batch, y_batch)
        g_norm = torch.norm(g)

        if g_norm < 1e-10:
            return torch.zeros_like(g), 0

        # Wrap HVP function to count calls
        def counted_hvp_fn(v):
            nonlocal hvp_count
            hvp_count += 1
            return hvp_fn(v)

        # Run Lanczos starting from g. Must use full re-orthogonalization for stability.
        # The starting vector is implicitly normalized within _run_lanczos.
        Q, T = self._run_lanczos(counted_hvp_fn, dim, k_max, v_start=g, reorthogonalize=True)

        if T.numel() == 0:
             print("Lanczos failed to run.")
             return torch.zeros_like(g), hvp_count

        # Solve the small projected problem
        y_star, lambda_star = self._solve_tridiagonal_subproblem(T, g_norm, M)

        # Project back to the original space: p* = Q y*
        # Q is a list of vectors, stack them into a matrix V_k
        Q_matrix = torch.stack(Q, dim=1)
        p_star = Q_matrix @ y_star

        end_time = time.time()
        print(f"Strategy 1 finished in {end_time - start_time:.4f}s. Total Hvps: {hvp_count}. Lambda*: {lambda_star:.4f}")
        return p_star, hvp_count

    # ========================================================================
    # Strategies 2 & 4: Nested Newton-CG (O(N) memory)
    # ========================================================================

    def _estimate_lambda_min(self, hvp_fn, dim, k_lanczos):
        """Estimates the minimum eigenvalue using Lanczos with a random start."""
        # We use reorthogonalization for stability in the estimation process, 
        # even though we don't store Q (if using the memory-efficient mode of _run_lanczos).
        # However, _run_lanczos implementation here stores Q internally if reorthogonalize=True.
        # For true O(N) memory estimation, we should use reorthogonalize=False or modify _run_lanczos.
        # Let's use reorthogonalize=False for the estimation, as extreme eigenvalues converge quickly even without it.
        T_est = self._run_lanczos(hvp_fn, dim, k_lanczos, v_start=None, reorthogonalize=False)
        
        if T_est.numel() == 0:
             return 0.0, k_lanczos
             
        # Use torch.linalg.eigvalsh for eigenvalues of symmetric matrix T
        eigenvalues = torch.linalg.eigvalsh(T_est)
        return eigenvalues[0].item(), k_lanczos

    def solve_nested_newton_cg(self, x_batch, y_batch, M, k_lanczos=20, inexact=False, max_newton_iters=50, cg_tol_fixed=1e-6, max_cg_iters=200):
        """
        Strategy 2 (inexact=False) and Strategy 4 (inexact=True). O(N) memory.
        """
        strategy_name = "4: Inexact Newton-CG (Warm Start)" if inexact else "2: Nested Newton-CG (Exact)"
        print(f"\nRunning Strategy {strategy_name} (M={M})")
        start_time = time.time()
        total_hvp_count = 0

        g, hvp_fn, dim = self._get_grad_and_hvp_fn(x_batch, y_batch)
        g_norm = torch.norm(g)

        if g_norm < 1e-10:
            return torch.zeros_like(g), 0

        # Step 1: Estimate lambda_min(H) using Lanczos (Required initialization for CG)
        lambda_min_est, hvps = self._estimate_lambda_min(hvp_fn, dim, k_lanczos)
        total_hvp_count += hvps
        # print(f"Estimated lambda_min: {lambda_min_est:.4f}")

        # Initialize lambda for the outer Newton loop
        # We need lambda > -lambda_min to ensure H+lambda I is PD for CG.
        safeguard = 1e-6
        lambda_boundary = max(0.0, -lambda_min_est)
        # Initialize slightly above the boundary
        lambda_k = lambda_boundary + safeguard

        # Initialize p_k for warm starting (Strategy 4)
        p_k = None 
        
        # Initialize tolerance for Strategy 4
        if inexact:
            # Heuristic: Start loose (e.g., related to the gradient norm).
            cg_tol = min(0.5, math.sqrt(g_norm.item()))
        else:
            cg_tol = cg_tol_fixed

        # Outer Newton Loop (Root finding for lambda)
        for i in range(max_newton_iters):

            # Define the shifted HVP function: A(v) = (H + lambda I)v
            def shifted_hvp_fn(v):
                return hvp_fn(v) + lambda_k * v

            # Inner CG Solve 1: (H + lambda I)p = -g

            # Strategy 4: Use current tolerance and warm start (p_k)
            # Strategy 2: Fixed tolerance and cold start (None)
            x0 = p_k if inexact else None
            
            p_k, hvps = self._cg_solve(shifted_hvp_fn, -g, x0=x0, tol=cg_tol, max_iters=max_cg_iters)
            total_hvp_count += hvps
            p_norm = torch.norm(p_k)

            # Handle potential issues if lambda_min was underestimated (H+lambda I is not PD)
            # If CG terminated due to non-positive curvature, p_k is the best estimate.
            # We check for catastrophic failure (though unlikely if CG is robust)
            if p_norm == 0 and g_norm > 0:
                 # print(f"Warning: CG potentially failed (p_norm=0). Increasing lambda.")
                 lambda_k = max(lambda_k * 2, lambda_k + safeguard)
                 p_k = None # Reset warm start
                 continue

            # Check convergence of the secular equation: Psi(lambda) = ||p|| - 2*lambda/M
            Psi = p_norm - (2 * lambda_k / M)
            # print(f"  Newton iter {i}: lambda={lambda_k:.4f}, Psi={Psi:.4e}, CG_tol={cg_tol:.1e}, Hvps={hvps}")

            # Convergence check (using relative tolerance)
            if abs(Psi) < 1e-5 * (1 + p_norm):
                # print("  Secular equation converged.")
                break

            # Newton update for lambda
            # Requires derivative d(Psi)/d(lambda) = -(p^T w) / ||p|| - 2/M
            # where w = (H+lambda I)^-1 p.

            # Inner CG Solve 2: (H+lambda I)w = p
            # We use the current cg_tol for the sensitivity solve as well.
            w_k, hvps = self._cg_solve(shifted_hvp_fn, p_k, x0=None, tol=cg_tol, max_iters=max_cg_iters)
            total_hvp_count += hvps

            p_dot_w = torch.dot(p_k, w_k)
            
            # Robustness check: p^T w must be positive if H+lambda I is PD.
            # p^T w = p^T (H+lambda I)^-1 p.
            if p_dot_w <= 1e-12 * p_norm**2:
                # print(f"Warning: p^T w <= 0 ({p_dot_w:.2e}). H+lambda I might not be PD. Increasing lambda.")
                lambda_k = max(lambda_k * 2, lambda_k + safeguard)
                p_k = None
                continue

            d_Psi = -(p_dot_w / p_norm) - (2 / M)

            # Newton step
            lambda_k_new = lambda_k - Psi / d_Psi

            # Safeguard lambda (ensure it doesn't violate PSD constraint estimate)
            lambda_k = max(lambda_k_new, lambda_boundary + safeguard)

            # Update tolerance for next iteration (Strategy 4)
            if inexact:
                # Heuristic: Tighten tolerance geometrically
                cg_tol = max(cg_tol_fixed, cg_tol * 0.1) 

        # Optional refinement step (Strategy 4): Run one final CG with tight tolerance
        if inexact and cg_tol > cg_tol_fixed:
            # print("Refining solution with high accuracy CG...")
            def final_shifted_hvp_fn(v):
                return hvp_fn(v) + lambda_k * v
            p_k, hvps = self._cg_solve(final_shifted_hvp_fn, -g, x0=p_k, tol=cg_tol_fixed, max_iters=max_cg_iters)
            total_hvp_count += hvps

        end_time = time.time()
        print(f"Strategy {strategy_name} finished in {end_time - start_time:.4f}s. Total Hvps: {total_hvp_count}. Lambda*: {lambda_k:.4f}")
        return p_k, total_hvp_count


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# END: Cubic Subproblem Solver Implementation
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Test Harness
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def run_solver_test(args):
    """
    Initializes model and runs the different cubic subproblem solvers for comparison.
    """
    if 'torch' not in sys.modules:
        print("PyTorch not available. Cannot run test.")
        return

    # Setup device
    if not torch.cuda.is_available() and 'cuda' in args.device:
        print("WARNING: CUDA not available. Switching to CPU.")
        device = torch.device("cpu")
    elif 'cuda' in args.device and torch.cuda.is_available():
        # Ensure the specific CUDA device exists
        if torch.cuda.device_count() > int(args.device.split(':')[-1]):
             device = torch.device(args.device)
        else:
             print(f"WARNING: Device {args.device} not found. Switching to default CUDA device.")
             device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    print(f"Testing on device: {device}")

    # Setup precision
    dtype_map = {
        "fp32": torch.float32,
        "fp64": torch.float64
    }
    if args.precision not in dtype_map:
         raise ValueError(f"Unknown precision: {args.precision}. Choose from 'fp32', 'fp64'.")
    precision_dtype = dtype_map[args.precision]
    print(f"Using precision: {args.precision} ({precision_dtype})")
    
    if args.precision == 'fp32':
         print("Note: fp64 is generally recommended for stability in second-order methods.")

    # 1. Initialize Model, Data, and Loss
    print(f"Initializing MLP(d={args.d}, h={args.h}, L={args.L})...")
    # Set seed for reproducibility
    torch.manual_seed(42)
    net = MLP(d=args.d, h=args.h, L=args.L, activation=args.activation).to(device=device, dtype=precision_dtype)
    
    print(f"Creating dummy data (n={args.n}, d={args.d})...")
    X = torch.randn(args.n, args.d, device=device, dtype=precision_dtype)
    # Generate binary labels based on a simple pattern to ensure a non-trivial Hessian
    y = (torch.sum(X**2, dim=1) > args.d/2).float().unsqueeze(1).to(device=device, dtype=precision_dtype)

    # Use a standard loss function
    criterion = nn.BCEWithLogitsLoss()

    solver = CubicSubproblemSolver(net, criterion, device=device)

    M = args.M # Cubic regularization parameter
    k_max = args.lanczos_k

    results = {}

    print("\n" + "="*60)

    # Run Strategy 1
    if "1" in args.strategies:
        if hasattr(torch.cuda, 'empty_cache'): torch.cuda.empty_cache()
        p1, hvp1 = solver.solve_direct_lanczos(X, y, M, k_max)
        results['S1 (Direct Lanczos)'] = (p1, hvp1)
        print("="*60)

    # Run Strategy 2
    if "2" in args.strategies:
        if hasattr(torch.cuda, 'empty_cache'): torch.cuda.empty_cache()
        # Use a small k_lanczos (e.g., 20) for initialization
        p2, hvp2 = solver.solve_nested_newton_cg(X, y, M, k_lanczos=20, inexact=False, max_cg_iters=args.max_cg)
        results['S2 (Newton-CG Exact)'] = (p2, hvp2)
        print("="*60)

    # Run Strategy 4
    if "4" in args.strategies:
        if hasattr(torch.cuda, 'empty_cache'): torch.cuda.empty_cache()
        p4, hvp4 = solver.solve_nested_newton_cg(X, y, M, k_lanczos=20, inexact=True, max_cg_iters=args.max_cg)
        results['S4 (Newton-CG Inexact)'] = (p4, hvp4)
        print("="*60)

    # Comparison
    if len(results) > 1:
        print("\n--- Comparison Report ---")
        
        # Use the first executed strategy as reference (usually S1 if available)
        ref_key = sorted(results.keys())[0]
        p_ref, hvp_ref = results[ref_key]
        norm_p_ref = torch.norm(p_ref)

        print(f"Reference: {ref_key}. HVPs: {hvp_ref}. Norm: {norm_p_ref:.6f}")

        for key in sorted(results.keys()):
            if key == ref_key: continue
            p, hvp = results[key]
            
            # Calculate relative error
            if norm_p_ref < 1e-9:
                rel_err = torch.norm(p_ref - p)
            else:
                rel_err = torch.norm(p_ref - p) / norm_p_ref
            
            # Format the relative error safely
            try:
                rel_err_val = rel_err.item()
            except AttributeError:
                rel_err_val = float(rel_err)

            print(f"Strategy: {key}. HVPs: {hvp}. Norm: {torch.norm(p):.6f}. Rel Error: {rel_err_val:.6e}")


if __name__ == "__main__":
    # Argument parser setup
    ap = argparse.ArgumentParser(description="Test Cubic Subproblem Solvers using Hvp-only methods")
    
    # --- Model Args (Smaller defaults for testing) ---
    ap.add_argument("--d", type=int, default=5, help="input dimension")
    ap.add_argument("--h", type=int, default=50, help="hidden layer dimension")
    ap.add_argument("--L", type=int, default=3, help="number of hidden layers")
    ap.add_argument("--activation", type=str, default="tanh",
                    choices=["relu", "gelu", "tanh"],
                    help="Activation function to use")
    
    # --- Data/Run Args ---
    ap.add_argument("--n", type=int, default=100, help="points per task (batch size)")
    ap.add_argument("--device", type=str, default=DEFAULT_DEVICE,
                    help=f"Device to run the test on (e.g., 'cuda:0', 'cpu'). Default: {DEFAULT_DEVICE}")
    ap.add_argument("--precision", type=str, default="fp64",
                    choices=["fp32", "fp64"],
                    help="Precision to use: fp32 (single) or fp64 (double). Default: fp64 (recommended)")

    # --- Solver Specific Args ---
    ap.add_argument("--M", type=float, default=1.0, help="Cubic regularization parameter (M)")
    ap.add_argument("--strategies", type=str, default="124", help="Strategies to run (e.g., '14' runs 1 and 4). Default: 124")
    ap.add_argument("--lanczos_k", type=int, default=50,
                    help="Max Lanczos iterations (k) for Strategy 1")
    ap.add_argument("--max_cg", type=int, default=200,
                    help="Max CG iterations for Strategies 2 & 4.")

    # Handle execution within environments where sys.argv might be complex or absent (e.g. notebooks)
    try:
        # A common way to check if running interactively (e.g., IPython/Jupyter)
        if 'ipykernel' in sys.modules:
            args = ap.parse_args([]) # Parse empty list to use defaults
        else:
            args = ap.parse_args()
    except (SystemExit, ImportError, NameError):
        # Fallback for environments where parsing might exit or required modules aren't fully loaded
        print("Parsing arguments failed or running in a restricted environment, attempting to use default arguments.")
        try:
            args = ap.parse_args([])
        except Exception as e:
            print(f"Could not even parse defaults: {e}")
            sys.exit(1)

    run_solver_test(args)