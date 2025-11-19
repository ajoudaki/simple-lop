import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, jvp, grad
import torchvision
import torchvision.transforms as transforms
import time

# ==============================================================================
# 1. Centralized Configuration
# ==============================================================================

CONFIG = {
    # System
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    
    # Dataset
    "dataset": "mnist", 
    "num_samples": 2000,
    "dummy_dim": 50,          
    
    # Model
    "hidden_dim": 1000,
    
    # Cubic Regularization (Adaptive)
    "M_init": 10.0,           # Starting M
    "M_min": 1e-4,           # Floor to prevent instability
    "M_max": 1e4,            # Ceiling to prevent stalling
    "rho_accept": 0.1,       # Threshold to accept a step
    "rho_great": 0.7,        # Threshold to decrease M (trust region expands)
    "scale_up": 2.0,         # Factor to increase M
    "scale_down": 0.5,       # Factor to decrease M
    
    # Inner Solver (Newton/MINRES)
    "newton_tol": 1e-3,
    "newton_max_iter": 50,   # Reduced slightly as adaptive M converges faster
    "minres_tol": 1e-4,
    "minres_max_iter": 1000,
    
    # Training
    "epochs": 20,
    "check_optimality": False, 
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
    except RuntimeError:
        dataset = dataset_cls(root='./data', train=True, download=True, transform=transforms.Compose(transform_list))
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=num_samples, shuffle=True)
    X, y = next(iter(loader))
    return X.view(num_samples, -1).to(device), y.to(device), input_dim, output_dim, 'cross_entropy'

# ==============================================================================
# 3. MINRES Solver
# ==============================================================================

def minres_solver(matvec_op, b, x_init=None, tol=1e-5, max_iter=100):
    def A_op(v):
        with torch.enable_grad(): return matvec_op(v)

    mv_count = 0
    with torch.no_grad():
        if x_init is not None:
            x = x_init.clone()
            r0 = b - A_op(x)
            mv_count += 1
        else:
            x = torch.zeros_like(b)
            r0 = b.clone()
            
        v = r0.clone()
        beta = torch.norm(v)
        if beta < 1e-20: return x, mv_count
            
        v.div_(beta)
        v_old, w, w_old = torch.zeros_like(v), torch.zeros_like(v), torch.zeros_like(v)
        c, s = 1.0, 0.0
        c_old, s_old = 1.0, 0.0
        beta_1 = beta.item()
        norm_b = max(torch.norm(b).item(), 1.0)
        
        for _ in range(max_iter):
            Av = A_op(v); mv_count += 1
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
            v_old.copy_(v)
            v.copy_(v_next.div_(beta_next))
            beta = beta_next
            w_old.copy_(w)
            w.copy_(d)
            c_old, s_old = c, s
            c, s = c_new, s_new
            
            if abs(beta_1) / norm_b < tol: break
                
    return x, mv_count

# ==============================================================================
# 4. Adaptive Cubic Step (ARC)
# ==============================================================================

def cubic_regularized_step_adaptive(model, params_flat, buffers, inputs, targets, config, current_M, loss_type='mse'):
    
    # --- Setup Loss Wrapper ---
    param_shapes = [p.shape for p in model.parameters()]
    param_numels = [p.numel() for p in model.parameters()]
    param_names = [n for n, _ in model.named_parameters()]
    
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
    grad_fn = grad(loss_wrt_params)
    grads = grad_fn(params_flat)
    
    def hvp(v):
        _, hv = jvp(grad_fn, (params_flat,), (v,))
        return hv

    # --- Newton Solve for Lambda (Subproblem) ---
    M = current_M
    g = grads
    lam = 0.0
    s = torch.zeros_like(g)
    total_mv = 0
    
    # Find step 's' that minimizes cubic model mk(s)
    for _ in range(config["newton_max_iter"]):
        def shifted_hvp(v): return hvp(v) + lam * v
        
        s, mv_s = minres_solver(shifted_hvp, -g, x_init=s, tol=config["minres_tol"], max_iter=config["minres_max_iter"])
        total_mv += mv_s
        
        norm_s = torch.norm(s)
        F_val = lam - (M / 2.0) * norm_s
        
        if abs(F_val) < config["newton_tol"]: break
            
        y, mv_y = minres_solver(shifted_hvp, s, x_init=None, tol=config["minres_tol"], max_iter=config["minres_max_iter"])
        total_mv += mv_y
        
        dot_sy = torch.dot(s, y)
        F_prime = 1.0 + (M / 2.0) * (dot_sy / (max(norm_s, 1e-8)))
        lam = max(0.0, float(lam - F_val / max(F_prime.item(), 1e-2)))
    
    # --- Adaptive Logic (The "Principled" Part) ---
    
    # 1. Predicted Reduction (Using 1 Extra HVP for accuracy)
    # Model change = - (g^T s + 0.5 s^T H s + M/6 ||s||^3)
    Hs = hvp(s)
    total_mv += 1
    
    term_lin = torch.dot(g, s)
    term_quad = 0.5 * torch.dot(s, Hs)
    term_cub = (M / 6.0) * torch.pow(torch.norm(s), 3)
    pred_reduction = -(term_lin + term_quad + term_cub)

    # 2. Actual Reduction
    new_params_candidate = params_flat + s
    with torch.no_grad():
        current_loss = loss_wrt_params(params_flat)
        new_loss_candidate = loss_wrt_params(new_params_candidate)
    actual_reduction = current_loss - new_loss_candidate
    
    # 3. Compute Rho
    # Add epsilon to denominator to avoid div/0 if prediction is tiny
    rho = actual_reduction / (pred_reduction + 1e-12)
    
    # 4. Update M and Accept/Reject
    step_accepted = False
    
    if rho >= config["rho_accept"]:
        # Step accepted
        final_params = new_params_candidate
        final_loss = new_loss_candidate
        step_norm = torch.norm(s).item()
        step_accepted = True
        
        # If model was extremely accurate, loosen the regularization (decrease M)
        if rho >= config["rho_great"]:
            current_M = max(config["M_min"], current_M * config["scale_down"])
    else:
        # Step rejected (Loss didn't go down enough, or went up)
        final_params = params_flat
        final_loss = current_loss
        step_norm = 0.0
        # M is not penalized enough, tighten it (increase M)
        current_M = min(config["M_max"], current_M * config["scale_up"])

    return final_params, final_loss, total_mv, step_norm, current_M, rho.item(), step_accepted

# ==============================================================================
# 5. Main Execution
# ==============================================================================

def main():
    print(f"Running on: {CONFIG['device']}")
    torch.manual_seed(CONFIG['seed'])
    if CONFIG['device'] == 'cuda': torch.cuda.manual_seed(CONFIG['seed'])
        
    X, y, in_dim, out_dim, loss_type = prepare_data(
        CONFIG['dataset'], CONFIG['num_samples'], CONFIG['device'], dummy_dim=CONFIG['dummy_dim']
    )
    
    model = nn.Sequential(
        nn.Linear(in_dim, CONFIG['hidden_dim']), nn.ReLU(),
        nn.Linear(CONFIG['hidden_dim'], CONFIG['hidden_dim']), nn.ReLU(),
        nn.Linear(CONFIG['hidden_dim'], CONFIG['hidden_dim']), nn.ReLU(),
        nn.Linear(CONFIG['hidden_dim'], out_dim)
    ).to(CONFIG['device'])
    
    print(f"Model: MLP {in_dim}->{CONFIG['hidden_dim']}x3->{out_dim} | Task: {loss_type.upper()}")
    
    flat_params = torch.cat([p.view(-1) for p in model.parameters()])
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
        
        # Call Adaptive Step
        new_params, loss, mvs, s_norm, next_M, rho, accepted = cubic_regularized_step_adaptive(
            model, flat_params, buffers, X, y, CONFIG, current_M, loss_type
        )
        
        dt = time.time() - start_t
        total_time += dt
        
        # Status marker: Check mark if accepted, 'x' if rejected
        status = "" if accepted else "(Rej)"
        
        print(f"{epoch:<3} | {loss.item():.5f}    | {current_M:.2e} | {rho:+.3f}   | {s_norm:.4f}   | {mvs:<4} | {dt:.2f}s {status}")
        
        # Update State
        flat_params = new_params
        current_M = next_M
        
        # Sync model for final evaluation or saving
        idx = 0
        with torch.no_grad():
            for p in model.parameters():
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