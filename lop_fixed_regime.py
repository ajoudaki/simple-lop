import numpy as np
import math
import matplotlib.pyplot as plt
import argparse
import sys
import time
import json
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.optim.optimizer import Optimizer

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# START: CORRECTED Muon Optimizer Implementation
# (Based on arXiv:2502.16982v1)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class Muon(Optimizer):
    """
    Implements the SCALABLE Muon optimizer as described in arXiv:2502.16982v1.

    This is a hybrid optimizer:
    - Applies Muon-P logic (Eq. 2 & 4) to 2D matrix parameters (e.g., weights).
    - Applies AdamW logic to all other parameters (e.g., biases, 1D vectors).
    """

    def __init__(self, params, lr=1e-3,
                 mu=0.95, k=5, rms_match_scale=0.2, weight_decay=0.1,
                 adam_betas=(0.9, 0.999), adam_eps=1e-8):
        
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= mu < 1.0:
            raise ValueError(f"Invalid Muon mu (momentum): {mu}")
        if not 1 <= k:
            raise ValueError(f"Invalid Muon K (iterations): {k}")
        if not 0.0 <= adam_betas[0] < 1.0:
            raise ValueError(f"Invalid Adam beta1: {adam_betas[0]}")
        if not 0.0 <= adam_betas[1] < 1.0:
            raise ValueError(f"Invalid Adam beta2: {adam_betas[1]}")
        if not 0.0 <= adam_eps:
            raise ValueError(f"Invalid Adam epsilon: {adam_eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        # Muon-specific Newton-Schulz coefficients from paper [cite: 98]
        self.a = 3.4445
        self.b = -4.7750
        self.c = 2.0315
            
        defaults = dict(
            lr=lr, mu=mu, k=k, rms_match_scale=rms_match_scale,
            weight_decay=weight_decay, adam_betas=adam_betas, adam_eps=adam_eps
        )
        super().__init__(params, defaults)

    def _newton_schulz_ortho(self, M, K):
        """
        Computes the orthogonalized matrix O_t = X_N directly from M.
        Implements Equation (2) from arXiv:2502.16982v1.
        """
        # Initialize X_0 = M_t / ||M_t||_F [cite: 92]
        M_norm = torch.norm(M, 'fro') + 1e-12 # Add eps for stability
        X = M / M_norm
        
        # Iterate K times 
        for _ in range(K):
            X_XT = X @ X.T
            # Equation (2): X_k = a*X + b*(X*X^T)*X + c*(X*X^T)^2*X 
            X = self.a * X + self.b * (X_XT @ X) + self.c * (X_XT @ X_XT @ X)
        
        return X # This is O_t

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # Get group-level hyperparameters
            lr = group['lr']
            wd = group['weight_decay']
            
            # Adam fallback params
            adam_beta1, adam_beta2 = group['adam_betas']
            adam_eps = group['adam_eps']
            
            # Muon params
            mu = group['mu']
            k = group['k']
            rms_scale = group['rms_match_scale']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Muon optimizer does not support sparse gradients.')

                state = self.state[p]

                # ==========================================================
                # Case 1: 2D Matrix -> Apply Scalable Muon Logic (Eq. 4)
                # ==========================================================
                if p.dim() == 2:
                    m, n = p.shape
                    
                    # Initialize Muon momentum state
                    if 'momentum' not in state:
                        state['momentum'] = torch.zeros_like(p)

                    M = state['momentum']
                    
                    # 1. Update Momentum (Eq. 1): M_t = mu*M_t-1 + G_t 
                    M.mul_(mu).add_(grad)

                    # 2. Compute Orthogonalized Update O_t (Eq. 2)
                    O_t = self._newton_schulz_ortho(M, k)
                    
                    # 3. Compute Final Scaled Update (Eq. 4)
                    # Get dim scale: sqrt(max(A,B)) 
                    dim_scale = math.sqrt(max(m, n))
                    
                    # W_t = W_t-1 - eta_t * (0.2 * O_t * sqrt(max(A,B)) + lambda * W_t-1) 
                    # We implement this with DECOUPLED weight decay, as inspired by AdamW [cite: 110]
                    
                    # 3a. Apply decoupled weight decay
                    if wd != 0:
                        p.add_(p, alpha=-lr * wd)
                    
                    # 3b. Apply scaled Muon update
                    # final_update = 0.2 * O_t * sqrt(max(A,B))
                    final_update = (rms_scale * dim_scale) * O_t
                    p.add_(final_update, alpha=-lr)

                # ==========================================================
                # Case 2: Other (e.g., 1D bias) -> Apply AdamW Logic
                # ==========================================================
                else:
                    # Initialize Adam state
                    if 'step' not in state:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)

                    state['step'] += 1
                    step = state['step']
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                    # AdamW: Decoupled Weight decay
                    if wd != 0:
                        p.add_(p, alpha=-lr * wd)

                    # Adam: m_t
                    exp_avg.mul_(adam_beta1).add_(grad, alpha=1.0 - adam_beta1)
                    # Adam: v_t
                    exp_avg_sq.mul_(adam_beta2).addcmul_(grad, grad, value=1.0 - adam_beta2)

                    # Bias correction
                    bias_correction1 = 1.0 - adam_beta1 ** step
                    bias_correction2 = 1.0 - adam_beta2 ** step
                    
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(adam_eps)
                    step_size = lr / bias_correction1
                    
                    # Apply update
                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# END: Muon Optimizer Implementation
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def effective_rank(Z):
    """
    Computes the effective rank of the covariance matrix of Z (pre-activations).
    Z shape: (batch_size, hidden_dim)
    eff_rank = exp(entropy(normalized_eigenvalues(Cov(Z))))
    """
    # Center the activations
    Z_c = Z - Z.mean(dim=0, keepdim=True)
    
    # Calculate covariance matrix
    # (B, D) -> (D, B) @ (B, D) -> (D, D)
    C = (Z_c.T @ Z_c) / (Z.shape[0] - 1)
    
    # Get real eigenvalues (since C is symmetric)
    eigvals = torch.linalg.eigvalsh(C)
    
    # Clamp small/negative eigenvalues for numerical stability
    eigvals = torch.clamp(eigvals, min=0)
    
    # Get sum of eigenvalues
    eigvals_sum = torch.sum(eigvals)
    
    # If all eigenvalues are zero, rank is 1 (or 0, but 1 is safer for exp(entropy))
    if eigvals_sum == 0:
        return torch.tensor(1.0, device=Z.device)
        
    # Normalize eigenvalues
    norm_eigvals = eigvals / eigvals_sum
    
    # Filter out zero eigenvalues to avoid log(0)
    nz_eigvals = norm_eigvals[norm_eigvals > 0]
    
    # Calculate entropy
    entropy = -torch.sum(nz_eigvals * torch.log(nz_eigvals))
    
    # Effective rank is exp(entropy)
    return torch.exp(entropy)


def fraction_duplicate_features(A, threshold=0.95):
    """
    Computes the fraction of duplicate features in post-activations (A).
    A shape: (batch_size, hidden_dim)
    A feature j is a duplicate if abs(corr(i, j)) > threshold for any i < j.
    """
    hidden_dim = A.shape[1]
    if hidden_dim == 0:
        return torch.tensor(0.0, device=A.device)
    
    # (D, B)
    A_T = A.T
    # (D, D)
    corr_matrix = torch.corrcoef(A_T)
    # Handle NaNs (e.g., from zero-variance features)
    corr_matrix = torch.nan_to_num(corr_matrix, nan=0.0)
    
    # Get absolute correlations in the upper-triangular part (i < j)
    upper_tri_abs = torch.abs(torch.triu(corr_matrix, diagonal=1))
    
    # Find features (columns j) that have at least one high correlation
    # with a feature i (row i) where i < j.
    # torch.any(dim=0) checks each column.
    is_duplicate_feature = torch.any(upper_tri_abs > threshold, dim=0)
    
    num_duplicates = torch.sum(is_duplicate_feature)
    
    return num_duplicates / hidden_dim


class MLP(nn.Module):
    """PyTorch MLP model"""
    def __init__(self, d=2, h=100, L=5, activation='tanh'):
        super().__init__()
        self.L = L # Number of hidden layers
        self.h = h # Hidden dimension

        # Set activation function
        activations = {
            'relu': nn.ReLU(),
            'selu': nn.SELU(),
            'gelu': nn.GELU(),
            'erf': lambda x: torch.erf(x),
            'tanh': nn.Tanh()
        }
        if activation.lower() not in activations:
            raise ValueError(f"Unknown activation: {activation}. Choose from {list(activations.keys())}")
        self.activation = activations[activation.lower()]
        self.activation_name = activation.lower()

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
                init.constant_(m.bias, 0)

    def activation_derivative(self, z):
        """Compute |f'(z)| for the activation function."""
        derivatives = {
            'relu': lambda x: (x > 0).double(),
            'selu': lambda x: torch.where(x > 0, torch.tensor(1.0507, device=x.device, dtype=x.dtype),
                                         1.0507 * 1.67326 * torch.exp(x)),
            'gelu': lambda x: torch.sigmoid(1.702 * x) * (1 + 1.702 * x * torch.sigmoid(1.702 * x) * (1 - torch.sigmoid(1.702 * x))),
            'erf': lambda x: (2.0 / math.sqrt(math.pi)) * torch.exp(-x**2),
            'tanh': lambda x: 1 - torch.tanh(x)**2
        }
        return torch.abs(derivatives[self.activation_name](z))

    def forward(self, X, capture=False):
        """
        Forward pass.
        If capture=True, returns intermediate pre- (Zs) and post- (As) activations.
        """
        if not capture:
            A = X
            for layer in self.layers:
                A = self.activation(layer(A))
            logits = self.output_layer(A)
            p = torch.sigmoid(logits)
            return p, logits
        else:
            Zs, As = [], []
            A = X
            for layer in self.layers:
                Z = layer(A)
                A = self.activation(Z)
                Zs.append(Z)
                As.append(A)
            logits = self.output_layer(A)
            p = torch.sigmoid(logits)
            return p, logits, Zs, As


def make_gmm_task(rng, n=96, d=2, k=4, delta=3.0, spread=0.8, sigma=0.22, relabel_p=0.0):
    """
    Generates a Gaussian Mixture Model task.
    This function remains in NumPy as it's data generation.
    """
    th = float(rng.uniform(0, 2 * math.pi))
    u = np.array([math.cos(th), math.sin(th)])
    shift = rng.normal(0, 0.6, size=d)
    a0 = -delta * u + shift[:2]
    a1 = delta * u + shift[:2]
    if d > 2:
        extra = rng.normal(0, 0.3, size=(2, d - 2))
        a0 = np.concatenate([a0, extra[0]])
        a1 = np.concatenate([a1, extra[1]])
    
    C0 = a0 + rng.normal(0, spread, size=(k, d))
    C1 = a1 + rng.normal(0, spread, size=(k, d))
    
    def draw(nh, C):
        idx = rng.integers(0, len(C), size=nh)
        mu = C[idx]
        return mu + rng.normal(0, sigma, size=(nh, d))
        
    nh = n // 2
    X = np.vstack([draw(nh, C0), draw(nh, C1)])
    y = np.vstack([np.zeros((nh, 1)), np.ones((nh, 1))])
    
    if relabel_p > 0.0:
        flip = (rng.random(size=(n, 1)) < relabel_p).astype(float)
        y = y * (1.0 - flip) + (1.0 - y) * flip
        
    perm = rng.permutation(n)
    # Return as float64 to match original
    return X[perm].astype(np.float64), y[perm].astype(np.float64)


def get_optimizer(model_params, args):
    """Helper function to create the selected optimizer."""
    
    opt_name = args.optimizer.lower()
    
    if opt_name == 'sgd':
        # Pass weight decay to SGD
        return optim.SGD(model_params, lr=args.lr, weight_decay=args.weight_decay)
        
    elif opt_name == 'adam':
        # Use AdamW (which is what 'Adam' usually means in this context)
        return optim.AdamW(model_params, lr=args.lr,
                           betas=args.adam_betas,
                           eps=args.adam_eps,
                           weight_decay=args.weight_decay)
                           
    elif opt_name == 'muon':
        # Use the new corrected Muon optimizer
        return Muon(model_params, lr=args.lr,
                    mu=args.muon_mu,
                    k=args.muon_k,
                    rms_match_scale=args.muon_rms_scale,
                    weight_decay=args.weight_decay,
                    adam_betas=args.adam_betas,
                    adam_eps=args.adam_eps)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")


def run(args):
    print(f"Using device: {device}, Optimizer: {args.optimizer}")
    
    # Set seeds for reproducibility
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # Use .double() to match original NumPy float64 precision
    net = MLP(d=args.d, h=args.h, L=args.L, activation=args.activation).to(device).double()
    
    # --- Create the selected optimizer ---
    optimizer = get_optimizer(net.parameters(), args)
    
    # Use stable BCEWithLogitsLoss (combines sigmoid + BCE)
    criterion = nn.BCEWithLogitsLoss()

    # Fixed training regime knobs
    steps = args.steps
    n = args.n
    # Use --fresh_steps if provided, otherwise default to --steps
    fresh_steps = args.fresh_steps if args.fresh_steps is not None else args.steps
    if fresh_steps != steps:
        print(f"Using {steps} steps for main tasks and {fresh_steps} for fresh baseline.")

    # Initialize metric lists
    min_losses = []
    final_losses = []
    gnorms = []
    # Use dicts to store layer-wise metrics
    eff_rank_data = {f'eff_rank_L{i}': [] for i in range(net.L)}
    dup_frac_data = {f'dup_frac_L{i}': [] for i in range(net.L)}
    frozen_frac_data = {f'frozen_frac_L{i}': [] for i in range(net.L)}

    # --- Main Experiment Loop (Single Phase) ---
    total_tasks = args.tasks
    print(f"Running {total_tasks} tasks...")
    
    for t in range(total_tasks):
        # Generate task using parameters from args
        X_np, y_np = make_gmm_task(rng, n=n, d=args.d, k=args.k,
                                   delta=args.delta, 
                                   spread=args.spread, 
                                   sigma=args.sigma, 
                                   relabel_p=args.relabel)
        
        X = torch.from_numpy(X_np).to(device).double()
        y = torch.from_numpy(y_np).to(device).double()
        
        min_ce = 1e9
        final_ce = 1e9
        g0 = None
        
        # Main task training loop
        for s in range(steps):
            optimizer.zero_grad()
            p, logits, Zs, As = net(X, capture=True)
            
            loss = criterion(logits, y)
            ce = loss.item()
            min_ce = min(min_ce, ce)
            if s == steps - 1: # Capture final loss
                final_ce = ce
                
            loss.backward()
            
            if s == 0:
                g0 = math.sqrt(sum(p.grad.norm(2).item()**2 
                                   for m in net.layers 
                                   for p in m.parameters() if p.grad is not None))
            
            optimizer.step()
        
        # --- Post-Training Metrics for this Task ---
        
        # Run a final forward pass to get final pre/post-activations
        with torch.no_grad():
            _, _, Zs_final, As_final = net(X, capture=True)

        # 1. Frozen Fraction (units with high probability of being frozen)
        for i, Z in enumerate(Zs_final):
            is_frozen = net.activation_derivative(Z) < args.frozen_thresh  # (batch, hidden_dim)
            frozen_prob = torch.mean(is_frozen.double(), dim=0)  # (hidden_dim,) - probability of being frozen
            is_frozen_unit = frozen_prob > args.frozen_p_thresh  # True if frozen on >= frozen_p_thresh of samples
            frozen_frac = torch.mean(is_frozen_unit.double()).item()
            frozen_frac_data[f'frozen_frac_L{i}'].append(frozen_frac)

        # 2. Effective Rank (on pre-activations Zs)
        for i, Z in enumerate(Zs_final):
            eff_rank = effective_rank(Z).item()
            eff_rank_data[f'eff_rank_L{i}'].append(eff_rank)

        # 3. Duplicate Feature Fraction (on post-activations As)
        for i, A in enumerate(As_final):
            dup_frac = fraction_duplicate_features(A, args.corr_thresh).item()
            dup_frac_data[f'dup_frac_L{i}'].append(dup_frac)

        # 4. Store other metrics
        gnorms.append(g0 if g0 is not None else 0.0)
        min_losses.append(min_ce)
        final_losses.append(final_ce)

        if (t + 1) % (max(1, total_tasks // 10)) == 0:
            print(f"  Task {t+1}/{total_tasks} completed.")


    # --- Optional fresh baseline ---
    fresh_mean = None
    fresh_median = None 
    
    if args.fresh_baseline > 0:
        print(f"Running {args.fresh_baseline} fresh baseline tasks (with {fresh_steps} steps)...")
        fresh_losses = []
        for t in range(args.fresh_baseline):
            # Generate task using the same parameters from args
            X_np, y_np = make_gmm_task(rng, n=n, d=args.d, k=args.k,
                                       delta=args.delta, 
                                       spread=args.spread, 
                                       sigma=args.sigma, 
                                       relabel_p=args.relabel)
            
            X = torch.from_numpy(X_np).to(device).double()
            y = torch.from_numpy(y_np).to(device).double()
            
            # Create a fresh model and optimizer
            f_net = MLP(d=args.d, h=args.h, L=args.L, activation=args.activation).to(device).double()
            # --- Create the selected optimizer for the fresh network ---
            f_opt = get_optimizer(f_net.parameters(), args)
            
            # Use fresh_steps for the fresh baseline
            for s in range(fresh_steps):
                f_opt.zero_grad()
                _, f_logits = f_net(X, capture=False)
                f_loss = criterion(f_logits, y)
                f_loss.backward()
                f_opt.step()
                
            _, final_logits = f_net(X, capture=False)
            final_ce_fresh = criterion(final_logits, y).item()
            fresh_losses.append(final_ce_fresh)
            
        fresh_mean = float(np.mean(fresh_losses))
        fresh_median = float(np.median(fresh_losses))

    # --- Results and Plots ---
    
    # Get stats for *all* tasks run in the main loop
    all_task_results = {
        "mean_min_CE": float(np.mean(min_losses)),
        "median_min_CE": float(np.median(min_losses)),
        "mean_final_CE": float(np.mean(final_losses)),
        "median_final_CE": float(np.median(final_losses)),
        "mean_hidden_grad_norm": float(np.mean(gnorms)),
        "fresh_mean_final_CE_on_late": fresh_mean,
        "fresh_median_final_CE_on_late": fresh_median,
    }

    # Add layer-wise metrics to results dict
    for i in range(net.L):
        all_task_results[f'mean_frozen_frac_L{i}'] = float(np.mean(frozen_frac_data[f'frozen_frac_L{i}']))
        all_task_results[f'median_frozen_frac_L{i}'] = float(np.median(frozen_frac_data[f'frozen_frac_L{i}']))
        all_task_results[f'mean_eff_rank_L{i}'] = float(np.mean(eff_rank_data[f'eff_rank_L{i}']))
        all_task_results[f'median_eff_rank_L{i}'] = float(np.median(eff_rank_data[f'eff_rank_L{i}']))
        all_task_results[f'mean_dup_frac_L{i}'] = float(np.mean(dup_frac_data[f'dup_frac_L{i}']))
        all_task_results[f'median_dup_frac_L{i}'] = float(np.median(dup_frac_data[f'dup_frac_L{i}']))
    
    print("\n--- Results (all tasks) ---")
    print(json.dumps(all_task_results, indent=2))

    # --- Save CSV with all tasks and new eff_rank columns ---
    T_total = args.tasks
    
    # Start with base metrics
    df_data = {
        "task": np.arange(1, T_total + 1),
        "min_CE": min_losses,
        "final_CE": final_losses,
        "grad_norm": gnorms
    }

    # Add layer-wise data to the dictionary
    for i in range(net.L):
        df_data[f'frozen_frac_L{i}'] = frozen_frac_data[f'frozen_frac_L{i}']
    for i in range(net.L):
        df_data[f'eff_rank_L{i}'] = eff_rank_data[f'eff_rank_L{i}']
    for i in range(net.L):
        df_data[f'dup_frac_L{i}'] = dup_frac_data[f'dup_frac_L{i}']
        
    # Create DataFrame
    df = pd.DataFrame(df_data)
    
    df.to_csv("out/lop_fixed_regime_results.csv", index=False)
    print(f"\nSaved all {T_total} task results to lop_fixed_regime_results.csv")

    # --- Save Plots (plotting all tasks) ---
    plt.style.use('ggplot') # Use a slightly nicer style for plots
    
    # Create a directory for plots if it doesn't exist
    import os
    os.makedirs("out", exist_ok=True)
    
    # Plot 1: min_CE
    fig1 = plt.figure(figsize=(10, 6))
    plt.plot(df["task"], df["min_CE"])
    plt.xlabel("Task")
    plt.ylabel("min CE")
    plt.title("Per-task min CE")
    plt.grid(True)
    fig1.savefig("out/lop_minCE.png", dpi=120)

    # Plot 2: Frozen Fraction (per layer)
    fig2 = plt.figure(figsize=(10, 6))
    ax = fig2.add_subplot(111)
    colors = plt.cm.coolwarm(np.linspace(0, 1, net.L))
    for i in range(net.L):
        ax.plot(df["task"], df[f'frozen_frac_L{i}'], label=f'Layer {i}', color=colors[i], alpha=0.8)
    ax.set_xlabel("Task")
    ax.set_ylabel(f"Frozen Fraction (|f'(z)| < {args.frozen_thresh} on >{args.frozen_p_thresh*100:.0f}% samples)")
    ax.set_title("Fraction of Frozen Units")
    ax.legend(title="Layer")
    ax.grid(True)
    ax.set_ylim(0, 1)
    fig2.savefig("out/lop_frozen.png", dpi=120)

    # Plot 3: Grad Norm
    fig3 = plt.figure(figsize=(10, 6))
    plt.plot(df["task"], df["grad_norm"])
    plt.xlabel("Task")
    plt.ylabel("||grad_hidden||")
    plt.title("Hidden grad norms (at step 0)")
    plt.grid(True)
    fig3.savefig("out/lop_grad.png", dpi=120) # Note: Original code saved this in root
    
    # Plot 4: final_CE
    fig4 = plt.figure(figsize=(10, 6))
    plt.plot(df["task"], df["final_CE"])
    plt.xlabel("Task")
    plt.ylabel("final CE")
    plt.title("Per-task final CE (at last step)")
    plt.grid(True)
    fig4.savefig("out/lop_finalCE.png", dpi=120)
    
    # Plot 5: Effective Rank
    fig5 = plt.figure(figsize=(10, 6))
    ax = fig5.add_subplot(111)
    colors = plt.cm.viridis(np.linspace(0, 1, net.L))
    for i in range(net.L):
        ax.plot(df["task"], df[f'eff_rank_L{i}'], label=f'Layer {i}', color=colors[i], alpha=0.8)
    
    ax.set_xlabel("Task")
    ax.set_ylabel("Effective Rank")
    ax.set_title("Effective Rank of Hidden Layer Pre-Activations")
    ax.legend(title="Layer")
    ax.grid(True)
    fig5.savefig("out/lop_effrank.png", dpi=120)
    
    # Plot 6: Duplicate Fraction
    fig6 = plt.figure(figsize=(10, 6))
    ax = fig6.add_subplot(111)
    colors = plt.cm.plasma(np.linspace(0, 1, net.L)) # Use a different colormap
    for i in range(net.L):
        ax.plot(df["task"], df[f'dup_frac_L{i}'], label=f'Layer {i}', color=colors[i], alpha=0.8)
    
    ax.set_xlabel("Task")
    ax.set_ylabel("Duplicate Feature Fraction")
    ax.set_title("Duplicate Features (Post-Activation)")
    ax.legend(title="Layer")
    ax.grid(True)
    ax.set_ylim(0, 1) # Fraction is always 0-1
    fig6.savefig("out/lop_dup_frac.png", dpi=120)
    
    print("Saved plots to 'out/' directory and 'lop_grad.png'")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    
    # --- General ---
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--d", type=int, default=4, help="input dimension")
    ap.add_argument("--h", type=int, default=100, help="hidden layer dimension")
    ap.add_argument("--L", type=int, default=5, help="number of hidden layers")
    ap.add_argument("--activation", type=str, default="tanh",
                    choices=["relu", "selu", "gelu", "erf", "tanh"],
                    help="Activation function to use")
    ap.add_argument("--frozen_thresh", type=float, default=0.05,
                    help="Threshold for frozen activations: |f'(z)| < threshold")
    ap.add_argument("--frozen_p_thresh", type=float, default=0.98,
                    help="Probability threshold: unit is frozen if frozen on > this fraction of samples")
    ap.add_argument("--n", type=int, default=96, help="points per task (batch size)")
    ap.add_argument("--k", type=int, default=4, help="number of Gaussian bumps")
    ap.add_argument("--lr", type=float, default=0.6, help="Fixed learning rate")
    ap.add_argument("--steps", type=int, default=40, help="fixed steps per task")
    ap.add_argument("--fresh_steps", type=int, default=None, help="Steps per task for fresh_baseline (defaults to --steps)")
    ap.add_argument("--tasks", type=int, default=100, help="Total number of tasks to run and log")
    ap.add_argument("--delta", type=float, default=2.8, help="Task cluster separation")
    ap.add_argument("--spread", type=float, default=0.7, help="Task cluster spread")
    ap.add_argument("--sigma", type=float, default=0.22, help="Task noise")
    ap.add_argument("--relabel", type=float, default=0.0, help="Label flip probability")
    ap.add_argument("--corr_thresh", type=float, default=0.95, help="Correlation threshold for duplicate features")
    ap.add_argument("--fresh_baseline", type=int, default=20, help="If >0, run a control group of fresh networks")
    
    # --- Optimizer Selection ---
    ap.add_argument("--optimizer", type=str, default="SGD",
                    choices=["SGD", "Adam", "Muon"],
                    help="Optimizer to use (SGD, Adam, Muon)")
    
    # --- Optimizer Hyperparameters (Shared) ---
    ap.add_argument("--weight_decay", type=float, default=0.0, 
                    help="Weight decay (lambda) for AdamW and Muon")

    # --- Muon-specific ---
    ap.add_argument("--muon_mu", type=float, default=0.95, 
                    help="Muon momentum mu ")
    ap.add_argument("--muon_k", type=int, default=5, 
                    help="Muon Newton-Schulz iterations (N) ")
    ap.add_argument("--muon_rms_scale", type=float, default=0.2, 
                    help="Muon RMS matching scale (e.g., 0.2) ")
    
    # --- Adam-specific (used by 'Adam' and 'Muon' fallback) ---
    ap.add_argument("--adam_betas", type=float, nargs=2, default=[0.9, 0.999], 
                    help="Adam betas")
    ap.add_argument("--adam_eps", type=float, default=1e-8, 
                    help="Adam epsilon")

    # These args are no longer used by the script but kept for compatibility
    ap.add_argument("--sat_bias", type=float, default=10.0)
    ap.add_argument("--sat_weight_scale", type=float, default=0.05)
    
    args = ap.parse_args()
    
    # A small fix from the original user code to ensure plots save to the 'out' dir
    import os
    os.makedirs("out", exist_ok=True)
    
    run(args)