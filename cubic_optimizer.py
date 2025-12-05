import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, jvp, grad
import torchvision
import torchvision.transforms as transforms
import time
import gc

# ==============================================================================
# 0. Diagnostic Utilities
# ==============================================================================

def format_bytes(size):
    # Helper to format bytes into MiB or GiB
    if size < 1024**3:
        return f"{size / (1024**2):.2f} MiB"
    else:
        return f"{size / (1024**3):.2f} GiB"

def memory_checkpoint(stage_name, snapshots, config, iteration=None, verbose=False):
    """Records the current and peak CUDA memory usage."""
    device = config.get("device", "cpu")
    if device != 'cuda' or not torch.cuda.is_available():
        return
    
    # Ensure all GPU operations are finished before measuring
    torch.cuda.synchronize()
    
    # Get current memory allocated and peak memory allocated since the last reset_peak_memory_stats()
    current_alloc = torch.cuda.memory_allocated()
    peak_alloc = torch.cuda.max_memory_allocated()
    
    key = stage_name
    if iteration is not None:
        key = f"[Iter {iteration}] {stage_name}"

    snapshots[key] = (current_alloc, peak_alloc)
    
    # Optional verbose printing for detailed debugging
    if verbose:
        print(f"  DEBUG: {key} - Current: {format_bytes(current_alloc)} | Peak: {format_bytes(peak_alloc)}")

# ==============================================================================
# 1. Centralized Configuration
# ==============================================================================

CONFIG = {
    # System
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    
    # Dataset
    "dataset": "cifar10", 
    "num_samples": 1000, # High batch size significantly increases HVP memory cost
    "dummy_dim": 50,          
    
    # Model
    "hidden_dim": 1000,
    
    # Cubic Regularization (Adaptive)
    "M_init": 10.0, "M_min": 1e-4, "M_max": 1e4,
    "rho_accept": 0.1, "rho_great": 0.7,
    "scale_up": 2, "scale_down": 0.7,
    
    # Inner Solver (Newton/MINRES)
    "newton_tol": 1e-2,
    "newton_max_iter": 50,
    "minres_tol": 1e-3,
    "minres_max_iter": 1000,
    
    # Training
    "epochs": 20,
    "check_optimality": False,
    # Enable detailed diagnostics for the first N epochs
    "audit_epochs": 1,
    # Enable highly detailed (verbose) HVP auditing inside MINRES
    "audit_verbose_hvp": True,
}

# ==============================================================================
# 2. Data Preparation
# ==============================================================================

def prepare_data(name, num_samples, device, dummy_dim=20):
    print(f"Preparing dataset: {name.upper()}...")
    name = name.lower()
    
    if name == 'dummy':
        input_dim, output_dim = dummy_dim, 1
        X = torch.randn(num_samples, input_dim, device=device)
        W_true = torch.randn(input_dim, 1, device=device)
        y = torch.sin(X @ W_true) + 0.05 * torch.randn(num_samples, 1, device=device)
        return X, y, input_dim, output_dim, 'mse'

    transform_list = [transforms.ToTensor()]
    if name == 'mnist':
        transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
        dataset_cls = torchvision.datasets.MNIST
        input_dim, output_dim = 28 * 28, 10
    elif name == 'cifar10':
        transform_list.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        dataset_cls = torchvision.datasets.CIFAR10
        input_dim, output_dim = 3 * 32 * 32, 10
    elif name == 'cifar100':
        transform_list.append(transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)))
        dataset_cls = torchvision.datasets.CIFAR100
        input_dim, output_dim = 3 * 32 * 32, 100
    else:
        raise ValueError(f"Unknown dataset: {name}")

    try:
        dataset = dataset_cls(root='./data', train=True, download=True, transform=transforms.Compose(transform_list))
    except Exception as e:
        print(f"Error loading dataset {name}: {e}. Falling back to dummy data.")
        return prepare_data('dummy', num_samples, device, dummy_dim)

    
    actual_num_samples = min(num_samples, len(dataset))
    loader = torch.utils.data.DataLoader(dataset, batch_size=actual_num_samples, shuffle=True)
    X, y = next(iter(loader))
    
    loss_t = 'cross_entropy' if output_dim > 1 else 'mse'
    return X.view(actual_num_samples, -1).to(device), y.to(device), input_dim, output_dim, loss_t

# ==============================================================================
# 3. MINRES Solver (Instrumented)
# ==============================================================================

def minres_solver(matvec_op, b, x_init=None, tol=1e-5, max_iter=100, snapshots=None, iteration=None, config=None):
    
    mv_count = 0
    # Determine if verbose logging is enabled in the config
    verbose = config.get("audit_verbose_hvp", False) if config else False

    # Instrumented A_op (HVP call)
    def A_op(v):
        nonlocal mv_count
        # Audit memory immediately before the HVP call
        if snapshots is not None:
             memory_checkpoint("MINRES: Before HVP", snapshots, config, iteration=f"{iteration}_mv_{mv_count}", verbose=verbose)
        
        result = matvec_op(v)
        mv_count += 1

        # Audit memory immediately after the HVP call. 
        # The 'Peak Alloc' recorded here will show the maximum memory used during the HVP execution.
        if snapshots is not None:
             memory_checkpoint("MINRES: After HVP", snapshots, config, iteration=f"{iteration}_mv_{mv_count}", verbose=verbose)
        return result

    # The entire solver runs without gradient tracking (as fixed previously).
    with torch.no_grad():
        if x_init is not None:
            x = x_init.clone()
            r0 = b - A_op(x)
        else:
            x = torch.zeros_like(b)
            r0 = b.clone()
            
        v = r0.clone()
        beta = torch.norm(v)
        if beta < 1e-20: return x, mv_count
            
        v.div_(beta)
        # Allocation of MINRES vectors (should be small)
        v_old, w, w_old = torch.zeros_like(v), torch.zeros_like(v), torch.zeros_like(v)
        c, s = 1.0, 0.0
        c_old, s_old = 1.0, 0.0
        beta_1 = beta.item()
        norm_b = max(torch.norm(b).item(), 1.0)
        
        for _ in range(max_iter):
            # The diagnostics are inside A_op()
            Av = A_op(v)
            
            # MINRES Algebra (Unchanged)
            alpha = torch.dot(v, Av)
            v_next = Av - alpha * v - beta * v_old
            beta_next = torch.norm(v_next)
            
            delta = c * alpha + s * beta
            gamma = s * alpha - c * beta
            epsilon = beta_next
            r1 = torch.sqrt(delta**2 + epsilon**2)
            
            if r1 < 1e-20: c_new, s_new = 1.0, 0.0
            else: c_new, s_new = delta / r1, epsilon / r1
                
            d = (v - gamma * w - s_old * beta * w_old) / r1
            x.add_(d, alpha=c_new * beta_1)
            
            beta_1 = -s_new * beta_1

            if abs(beta_1) / norm_b < tol: break

            v_old.copy_(v)

            if beta_next > 1e-20:
                v.copy_(v_next.div_(beta_next))
            else:
                break # Lucky breakdown or stagnation

            beta = beta_next
            w_old.copy_(w)
            w.copy_(d)
            c_old, s_old = c, s
            c, s = c_new, s_new
                            
    return x, mv_count

# ==============================================================================
# 4. Adaptive Cubic Step (ARC) (Instrumented)
# ==============================================================================

def cubic_regularized_step_adaptive(model, params_flat, buffers, inputs, targets, config, current_M, loss_type='mse', snapshots=None):
    
    if snapshots is not None:
        memory_checkpoint("Start of Step", snapshots, config)

    # --- Setup Loss Wrapper ---
    param_shapes = [p.shape for p in model.parameters() if p.requires_grad]
    param_numels = [p.numel() for p in model.parameters() if p.requires_grad]
    param_names = [n for n, p in model.named_parameters() if p.requires_grad]
    
    def loss_wrt_params(p_flat):
        params_dict = {}
        idx = 0
        for i, name in enumerate(param_names):
            n = param_numels[i]
            params_dict[name] = p_flat[idx : idx + n].view(param_shapes[i])
            idx += n
        out = functional_call(model, (params_dict, buffers), (inputs,))
        return F.mse_loss(out, targets) if loss_type == 'mse' else F.cross_entropy(out, targets)

    # --- Gradient & HVP ---
    if snapshots is not None:
        memory_checkpoint("Before Grad Calc", snapshots, config)

    grad_fn = grad(loss_wrt_params)
    # This executes the forward and backward pass (First-order)
    grads = grad_fn(params_flat)
    
    if snapshots is not None:
        # This shows memory usage including activations and the gradient tensor
        memory_checkpoint("After Grad Calc", snapshots, config)

    # Define the HVP function (Second-order: Forward-over-Reverse AD)
    def hvp(v):
        v_detached = v.detach()
        # This is the operation expected to cause the memory spike.
        _, hv = jvp(grad_fn, (params_flat,), (v_detached,))
        return hv

    # --- Newton Solve for Lambda (Subproblem) ---
    M = current_M
    g = grads
    lam = 0.0
    s = torch.zeros_like(g)
    total_mv = 0
    
    # Find step 's' that minimizes cubic model mk(s)
    for i in range(config["newton_max_iter"]):
        
        if snapshots is not None:
            memory_checkpoint("Start Newton Iter", snapshots, config, iteration=i)

        def shifted_hvp(v): return hvp(v) + lam * v
        
        # First MINRES call (Solving for s)
        # We pass the snapshots dictionary and config to enable internal auditing.
        s, mv_s = minres_solver(shifted_hvp, -g, x_init=s, tol=config["minres_tol"], max_iter=config["minres_max_iter"], snapshots=snapshots, iteration=f"{i}_S", config=config)
        total_mv += mv_s
        
        if snapshots is not None:
            memory_checkpoint("After Solve S", snapshots, config, iteration=i)

        # Newton root finding (Unchanged)
        with torch.no_grad():
            norm_s = torch.norm(s)
            F_val = lam - (M / 2.0) * norm_s
        
        if abs(F_val) < config["newton_tol"]: break
            
        # Second MINRES call (Solving for y)
        y, mv_y = minres_solver(shifted_hvp, s, x_init=None, tol=config["minres_tol"], max_iter=config["minres_max_iter"], snapshots=snapshots, iteration=f"{i}_Y", config=config)
        total_mv += mv_y
        
        if snapshots is not None:
            memory_checkpoint("After Solve Y", snapshots, config, iteration=i)

        # Lambda update (Unchanged)
        with torch.no_grad():
            dot_sy = torch.dot(s, y)
            F_prime = 1.0 + (M / 2.0) * (dot_sy / (max(norm_s, 1e-8)))
            lam = max(0.0, float(lam - F_val / max(F_prime.item(), 1e-2)))
    
    # --- Adaptive Logic ---
    # This section is correctly wrapped in no_grad()
    with torch.no_grad():
        # 1. Predicted Reduction (Final HVP call)
        if snapshots is not None:
            memory_checkpoint("Before Final HVP (Rho)", snapshots, config)
            
        Hs = hvp(s)
        total_mv += 1
        
        if snapshots is not None:
            # This checks the peak usage during the final HVP call
            memory_checkpoint("After Final HVP (Rho)", snapshots, config)

        # Calculations for adaptation (Unchanged)
        term_lin = torch.dot(g, s)
        term_quad = 0.5 * torch.dot(s, Hs)
        term_cub = (M / 6.0) * torch.pow(torch.norm(s), 3)
        pred_reduction = -(term_lin + term_quad + term_cub)

        # 2. Actual Reduction
        new_params_candidate = params_flat + s
   
        current_loss = loss_wrt_params(params_flat) 
        new_loss_candidate = loss_wrt_params(new_params_candidate)
        actual_reduction = current_loss - new_loss_candidate
        
        # 3. Compute Rho
        rho = actual_reduction / (pred_reduction + 1e-12)
    
        # 4. Update M and Accept/Reject (Unchanged)
        step_accepted = False
        
        if rho >= config["rho_accept"]:
            final_params = new_params_candidate
            final_loss = new_loss_candidate
            step_norm = torch.norm(s).item()
            step_accepted = True
            
            if rho >= config["rho_great"]:
                current_M = max(config["M_min"], current_M * config["scale_down"])
        else:
            final_params = params_flat
            final_loss = current_loss
            step_norm = 0.0
            current_M = min(config["M_max"], current_M * config["scale_up"])

    if snapshots is not None:
        memory_checkpoint("End of Step", snapshots, config)

    return final_params, final_loss, total_mv, step_norm, current_M, rho.item(), step_accepted

# ==============================================================================
# 5. Main Execution (With Diagnostics Reporting)
# ==============================================================================

def main():
    print(f"Running on: {CONFIG['device']}")
    if CONFIG['device'] == 'cuda' and torch.cuda.is_available():
        torch.cuda.init()
        print(f"CUDA Memory initialized: {format_bytes(torch.cuda.memory_allocated())}")
        torch.cuda.manual_seed(CONFIG['seed'])
    elif CONFIG['device'] == 'cuda':
        print("Warning: CUDA requested but not available.")

    torch.manual_seed(CONFIG['seed'])
        
    X, y, in_dim, out_dim, loss_type = prepare_data(
        CONFIG['dataset'], CONFIG['num_samples'], CONFIG['device'], dummy_dim=CONFIG['dummy_dim']
    )
    
    # Model definition (4 hidden layers)
    model = nn.Sequential(
        nn.Linear(in_dim, CONFIG['hidden_dim']), nn.ReLU(),
        nn.Linear(CONFIG['hidden_dim'], CONFIG['hidden_dim']), nn.ReLU(),
        nn.Linear(CONFIG['hidden_dim'], CONFIG['hidden_dim']), nn.ReLU(),
        nn.Linear(CONFIG['hidden_dim'], CONFIG['hidden_dim']), nn.ReLU(),
        nn.Linear(CONFIG['hidden_dim'], out_dim)
    ).to(CONFIG['device'])
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: MLP {in_dim}->{CONFIG['hidden_dim']}x4->{out_dim} | Task: {loss_type.upper()}")
    print(f"Total Parameters: {num_params:,} | Est. Model Size: {format_bytes(num_params * 4)}")


    # Initialize parameters as detached.
    flat_params = torch.cat([p.view(-1) for p in model.parameters() if p.requires_grad]).detach()
    buffers = dict(model.named_buffers())
    current_M = CONFIG["M_init"]
    
    # Formatted Header
    header = f"{'Ep':<3} | {'Loss':<10} | {'M':<8} | {'Rho':<7} | {'StepNorm':<8} | {'MVs':<4} | {'Time'}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    
    total_time = 0
    
    for epoch in range(CONFIG['epochs']):
        start_t = time.time()
        
        # --- DIAGNOSTICS SETUP ---
        snapshots = {}
        should_audit = epoch < CONFIG["audit_epochs"]
        is_cuda = CONFIG['device'] == 'cuda' and torch.cuda.is_available()
        
        if should_audit and is_cuda:
            print(f"\n*** Starting Memory Audit for Epoch {epoch} ***")
            # Clean up garbage and clear cache before the epoch starts for cleaner measurements
            gc.collect()
            torch.cuda.empty_cache()
            # Crucial: Reset peak stats to measure the peak usage of this specific epoch/step
            torch.cuda.reset_peak_memory_stats()
        # -------------------------

        # Call Adaptive Step
        new_params, loss, mvs, s_norm, next_M, rho, accepted = cubic_regularized_step_adaptive(
            model, flat_params, buffers, X, y, CONFIG, current_M, loss_type, snapshots=snapshots if (should_audit and is_cuda) else None
        )
        
        dt = time.time() - start_t
        total_time += dt
        
        status = "" if accepted else "(Rej)"
        
        print(f"{epoch:<3} | {loss.item():.5f}    | {current_M:.2e} | {rho:+.3f}   | {s_norm:.4f}    | {mvs:<4} | {dt:.2f}s {status}")
        
        # Update State (Detached)
        flat_params = new_params.detach()
        current_M = next_M

        # --- DIAGNOSTICS REPORTING ---
        if snapshots:
            print(f"\n--- Memory Diagnostics Report (Epoch {epoch}) ---")
            print(f"{'Stage':<45} | {'Current Alloc':<15} | {'Peak Alloc':<15}")
            print("-" * 80)
            
            # Determine if we should show the detailed HVP logs based on config
            verbose_hvp = CONFIG.get("audit_verbose_hvp", False)
            max_peak = 0
            
            for stage, (current, peak) in snapshots.items():
                max_peak = max(max_peak, peak)
                # Filter the internal MINRES HVP steps if not verbose
                if not verbose_hvp and "MINRES: Before HVP" in stage: continue
                if not verbose_hvp and "MINRES: After HVP" in stage: continue
                
                print(f"{stage:<45} | {format_bytes(current):<15} | {format_bytes(peak):<15}")
            
            print("-" * 80)
            print(f"Overall Peak Usage during step: {format_bytes(max_peak)}")
            print("-" * 80 + "\n")
        # -----------------------------


    # Sync model for final evaluation
    idx = 0
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                n = p.numel()
                p.copy_(flat_params[idx:idx+n].view(p.shape))
                idx += n

    # Final Accuracy Check
    if loss_type == 'cross_entropy':
        with torch.no_grad():
            acc = (torch.argmax(model(X), dim=1) == y).float().mean()
            print("-" * len(header))
            print(f"Final Training Accuracy: {acc.item()*100:.2f}%")

if __name__ == "__main__":
    main()