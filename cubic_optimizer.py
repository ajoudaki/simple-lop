import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, jvp, grad
import time
import math

# ==============================================================================
# 1. Centralized Configuration
# ==============================================================================

CONFIG = {
    # System
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dtype": torch.float32,
    "seed": 42,
    
    # Model Architecture
    "input_dim": 20,
    "hidden_dim": 1000,  # Increased as per your request
    "output_dim": 1,
    "num_samples": 1000,
    
    # Cubic Regularization Hyperparameters
    "M": 10.0,
    "newton_tol": 1e-2,
    "newton_max_iter": 20,
    
    # MINRES Linear Solver Hyperparameters
    "minres_tol": 1e-2,
    "minres_max_iter": 200,
    
    # Training Loop
    "epochs": 15,
    "check_optimality": True,
}

# ==============================================================================
# 2. MINRES Solver (Optimized for Memory)
# ==============================================================================

def minres_solver(matvec_op, b, x_init=None, tol=1e-5, max_iter=100):
    """
    Solves Ax = b using Matrix-Vector products.
    Critically: Uses torch.no_grad() for vector updates to prevent graph explosion.
    """
    # Wrap the operator to ensure it runs with autograd enabled, 
    # while the rest of MINRES runs in no_grad mode.
    def A_op(v):
        with torch.enable_grad():
            return matvec_op(v)

    mv_count = 0
    n = b.numel()
    
    # Everything inside MINRES logic (linear algebra) should not build a graph
    with torch.no_grad():
        if x_init is not None:
            x = x_init.clone()
            r0 = b - A_op(x) # A_op will re-enable grad internally
            mv_count += 1
        else:
            x = torch.zeros_like(b)
            r0 = b.clone()
            
        v = r0.clone()
        beta = torch.norm(v)
        
        if beta < 1e-20:
            return x, mv_count
            
        v = v / beta
        v_old = torch.zeros_like(v)
        w = torch.zeros_like(v)
        w_old = torch.zeros_like(v)
        
        c, s = 1.0, 0.0
        c_old, s_old = 1.0, 0.0
        
        beta_1 = beta.item()
        norm_b = torch.norm(b).item()
        if norm_b == 0: norm_b = 1.0
        
        for k in range(max_iter):
            # --- The only part that needs Autograd logic (internal to the func) ---
            Av = A_op(v) 
            mv_count += 1
            
            # --- The rest is pure linear algebra (no graph needed) ---
            alpha = torch.dot(v, Av)
            v_next = Av - alpha * v - beta * v_old
            beta_next = torch.norm(v_next)
            
            delta = c * alpha + s * beta
            gamma = s * alpha - c * beta
            epsilon = beta_next
            
            r1 = torch.sqrt(delta**2 + epsilon**2)
            
            if r1 < 1e-20: c_new, s_new = 1.0, 0.0
            else: c_new, s_new = delta / r1, epsilon / r1
                
            eta = c_new * beta_1
            d = (v - gamma * w - s_old * beta * w_old) / r1
            
            # Update x
            x.add_(d, alpha=eta)
            
            # Update scalars
            beta_1 = -s_new * beta_1
            
            # Shift vectors
            v_old.copy_(v)
            v.copy_(v_next.div_(beta_next))
            beta = beta_next
            
            w_old.copy_(w)
            w.copy_(d)
            c_old, s_old = c, s
            c, s = c_new, s_new
            
            if abs(beta_1) / norm_b < tol:
                break
                
    return x, mv_count

# ==============================================================================
# 3. Optimality Checker
# ==============================================================================

def check_optimality(H_op_func, g, s_star, lam_star, M):
    print("\n   [Optimality Check]")
    with torch.no_grad():
        # 1. Stationarity
        Hs = H_op_func(s_star)
        shifted_lhs = Hs + lam_star * s_star
        residual = torch.norm(shifted_lhs + g)
        print(f"   1. Res Norm ||(H+lamI)s + g|| : {residual:.2e}")

        # 2. Norm Condition
        norm_s = torch.norm(s_star)
        target_lam = (M / 2.0) * norm_s
        lam_err = abs(lam_star - target_lam)
        print(f"   2. Lam Check |lam - M/2||s||| : {lam_err:.2e}")

        # 3. PSD Check (Approximate)
        print(f"   3. PSD Check (Estimating lambda_min)...")
        min_observed = float('inf')
        for _ in range(5):
            v = torch.randn_like(g)
            v /= torch.norm(v)
            Hv = H_op_func(v)
            val = torch.dot(v, Hv) + lam_star
            if val < min_observed: min_observed = val.item()
                
        status = "PASS" if min_observed >= -1e-3 else "FAIL"
        print(f"      Min Rayleigh Quote (approx) : {min_observed:.4e} -> {status}")

# ==============================================================================
# 4. Cubic Step Solver (Stateless & Memory Efficient)
# ==============================================================================

def cubic_regularized_step(model, params_flat, buffers, inputs, targets, config):
    
    # --- 1. Define Stateless Functional Loss ---
    # We pass params and buffers explicitly to functional_call
    # This is the cleanest way to avoid memory leaks with torch.func
    
    # Pre-compute shapes once to avoid doing it in the loop
    param_shapes = [p.shape for p in model.parameters()]
    param_numels = [p.numel() for p in model.parameters()]
    param_names = [n for n, _ in model.named_parameters()]
    
    def compute_loss_stateless(p_flat, b_dict, x, y):
        # Unflatten
        params_dict = {}
        idx = 0
        for i, name in enumerate(param_names):
            length = param_numels[i]
            params_dict[name] = p_flat[idx : idx + length].view(param_shapes[i])
            idx += length
            
        # Functional forward pass
        out = functional_call(model, (params_dict, b_dict), (x,))
        return F.mse_loss(out, y)

    # --- 2. Gradient & HVP Setup ---
    
    # Compute Gradient
    # grad(func)(args) -> returns gradients with respect to arg 0 (p_flat)
    grad_fn = grad(compute_loss_stateless, argnums=0)
    grads = grad_fn(params_flat, buffers, inputs, targets)
    
    # Define HVP using Forward-over-Reverse (JVP of Grad)
    # This is the most efficient HVP method in PyTorch
    def hvp(v):
        # jvp(func, primals, tangents)
        # We only want derivative w.r.t params_flat (first arg), so other tangents are zeros/None
        # However, jvp expects a tangent for every primal argument.
        
        tangents = (v, {}, torch.zeros_like(inputs), torch.zeros_like(targets))
        primals = (params_flat, buffers, inputs, targets)
        
        # jvp returns (output, output_tangent). We want output_tangent.
        # The 'output' of grad_fn is the gradient. 
        # The 'output_tangent' of grad_fn is H*v.
        _, hv = jvp(grad_fn, primals, tangents)
        return hv

    # --- 3. Cubic Subproblem (Newton on Lambda) ---
    
    M = config["M"]
    g = grads
    
    lam = 0.0
    s = torch.zeros_like(g)
    total_mv = 0
    
    for i in range(config["newton_max_iter"]):
        
        def shifted_hvp(v):
            return hvp(v) + lam * v
        
        # Solve (H + lam*I) s = -g
        s, mv_s = minres_solver(shifted_hvp, -g, x_init=s, 
                                tol=config["minres_tol"], 
                                max_iter=config["minres_max_iter"])
        total_mv += mv_s
        
        norm_s = torch.norm(s)
        F_val = lam - (M / 2.0) * norm_s
        
        if abs(F_val) < config["newton_tol"]:
            break
            
        # Solve (H + lam*I) y = s
        y, mv_y = minres_solver(shifted_hvp, s, x_init=None, 
                                tol=config["minres_tol"], 
                                max_iter=config["minres_max_iter"])
        total_mv += mv_y
        
        dot_sy = torch.dot(s, y)
        term = (M / 2.0) * (dot_sy / (norm_s + 1e-8))
        F_prime = 1.0 + term
        
        denom = F_prime if F_prime > 1e-2 else 1.0
        lam_new = lam - (F_val / denom)
        
        if isinstance(lam_new, torch.Tensor): lam_new = lam_new.item()
        lam = max(0.0, lam_new)
    
    # --- 4. Diagnostics & Update ---
    if config["check_optimality"]:
        check_optimality(hvp, g, s, lam, M)
        
    new_params_flat = params_flat + s
    
    # Re-evaluate loss for reporting (cheap forward pass)
    with torch.no_grad():
        loss_val = compute_loss_stateless(params_flat, buffers, inputs, targets)
    
    return new_params_flat, loss_val, total_mv

# ==============================================================================
# 5. Main Execution
# ==============================================================================

def main():
    print(f"Running on: {CONFIG['device']}")
    torch.manual_seed(CONFIG['seed'])
    if CONFIG['device'] == 'cuda':
        torch.cuda.manual_seed(CONFIG['seed'])
        
    # 1. Setup Data
    X = torch.randn(CONFIG['num_samples'], CONFIG['input_dim']).to(CONFIG['device'])
    W_true = torch.randn(CONFIG['input_dim'], 1).to(CONFIG['device'])
    Y = torch.sin(X @ W_true) + 0.05 * torch.randn(CONFIG['num_samples'], 1).to(CONFIG['device'])
    
    # 2. Model
    model = nn.Sequential(
        nn.Linear(CONFIG['input_dim'], CONFIG['hidden_dim']),
        nn.ReLU(),
        nn.Linear(CONFIG['hidden_dim'], CONFIG['hidden_dim']),
        nn.ReLU(),
        nn.Linear(CONFIG['hidden_dim'], CONFIG['output_dim'])
    ).to(CONFIG['device'])
    
    # Flatten Params & Extract Buffers
    params_list = list(model.parameters())
    flat_params = torch.cat([p.view(-1) for p in params_list])
    buffers = dict(model.named_buffers())
    
    print(f"{'Epoch':<5} | {'Loss':<12} | {'MatVecs':<8} | {'Time (s)'}")
    print("-" * 45)
    
    total_time = 0
    
    for epoch in range(CONFIG['epochs']):
        start_t = time.time()
        
        # Perform Step
        new_flat_params, loss, mvs = cubic_regularized_step(
            model, flat_params, buffers, X, Y, CONFIG
        )
        
        flat_params = new_flat_params
        
        # Update PyTorch model params (for next iteration state tracking)
        idx = 0
        with torch.no_grad():
            for p in model.parameters():
                numel = p.numel()
                p.copy_(flat_params[idx:idx+numel].view(p.shape))
                idx += numel
        
        end_t = time.time()
        dt = end_t - start_t
        total_time += dt
        
        print(f"{epoch:<5} | {loss.item():.6f}     | {mvs:<8} | {dt:.4f}")

if __name__ == "__main__":
    main()