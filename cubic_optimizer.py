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
    
    # Dataset Selection
    # Options: 'dummy', 'mnist', 'cifar10', 'cifar100'
    "dataset": "mnist", 
    "num_samples": 2000,      # Batch size
    "dummy_dim": 50,          
    
    # Model Architecture
    "hidden_dim": 1000,
    
    # Cubic Regularization Hyperparameters
    "M": 5.0,                 # Initial M (adaptive strategies exist, but fixed is fine for demo)
    "newton_tol": 1e-3,
    "newton_max_iter": 100,
    
    # MINRES Linear Solver Hyperparameters
    "minres_tol": 1e-4,
    "minres_max_iter": 1000,
    
    # Training Loop
    "epochs": 20,
    "check_optimality": False, 
}

# ==============================================================================
# 2. Data Preparation Function
# ==============================================================================

def prepare_data(name, num_samples, device, dummy_dim=20):
    print(f"Preparing dataset: {name.upper()}...")
    name = name.lower()
    
    if name == 'dummy':
        input_dim = dummy_dim
        output_dim = 1
        X = torch.randn(num_samples, input_dim, device=device)
        W_true = torch.randn(input_dim, 1, device=device)
        y = torch.sin(X @ W_true) + 0.05 * torch.randn(num_samples, 1, device=device)
        return X, y, input_dim, output_dim, 'mse'

    # Transforms
    transform_list = [transforms.ToTensor()]
    
    if name == 'mnist':
        transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
        dataset_cls = torchvision.datasets.MNIST
        input_dim = 28 * 28
        output_dim = 10
    elif name == 'cifar10':
        transform_list.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        dataset_cls = torchvision.datasets.CIFAR10
        input_dim = 3 * 32 * 32
        output_dim = 10
    elif name == 'cifar100':
        transform_list.append(transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)))
        dataset_cls = torchvision.datasets.CIFAR100
        input_dim = 3 * 32 * 32
        output_dim = 100
    else:
        raise ValueError(f"Unknown dataset: {name}")

    transform = transforms.Compose(transform_list)
    
    # Load Dataset (Retry mechanism for stability)
    try:
        dataset = dataset_cls(root='./data', train=True, download=True, transform=transform)
    except RuntimeError:
        dataset = dataset_cls(root='./data', train=True, download=True, transform=transform)
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=num_samples, shuffle=True)
    iter_loader = iter(loader)
    X, y = next(iter_loader)
    
    # Flatten X
    X = X.view(num_samples, -1).to(device)
    y = y.to(device)
    
    return X, y, input_dim, output_dim, 'cross_entropy'

# ==============================================================================
# 3. MINRES Solver (Memory Optimized)
# ==============================================================================

def minres_solver(matvec_op, b, x_init=None, tol=1e-5, max_iter=100):
    # Wrap operator to re-enable gradients only for the HVP calculation
    def A_op(v):
        with torch.enable_grad():
            return matvec_op(v)

    mv_count = 0
    
    # Run the solver logic without tracking gradients for the iterative updates
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
            # HVP Call (Autograd enabled inside A_op)
            Av = A_op(v) 
            mv_count += 1
            
            # Standard MINRES steps
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
            x.add_(d, alpha=eta)
            
            beta_1 = -s_new * beta_1
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
# 4. Optimality Check (Optional)
# ==============================================================================

def check_optimality(H_op_func, g, s_star, lam_star, M):
    print("\n   [Optimality Check]")
    with torch.no_grad():
        Hs = H_op_func(s_star)
        res = torch.norm(Hs + lam_star * s_star + g)
        norm_s = torch.norm(s_star)
        lam_diff = abs(lam_star - (M/2.0)*norm_s)
        print(f"   1. Residual: {res:.2e} | 2. Lambda Check: {lam_diff:.2e}")

# ==============================================================================
# 5. Cubic Regularization Step
# ==============================================================================

def cubic_regularized_step(model, params_flat, buffers, inputs, targets, config, loss_type='mse'):
    
    # --- 1. Setup Stateless Loss Wrapper ---
    # We capture inputs, targets, and buffers in the closure so the function
    # only takes 'p_flat' as an argument. This fixes the JVP tangent issue.
    
    param_shapes = [p.shape for p in model.parameters()]
    param_numels = [p.numel() for p in model.parameters()]
    param_names = [n for n, _ in model.named_parameters()]
    
    def loss_wrt_params(p_flat):
        # Reconstruct parameters from flat vector
        params_dict = {}
        idx = 0
        for i, name in enumerate(param_names):
            length = param_numels[i]
            params_dict[name] = p_flat[idx : idx + length].view(param_shapes[i])
            idx += length
            
        # Functional forward pass
        out = functional_call(model, (params_dict, buffers), (inputs,))
        
        if loss_type == 'mse':
            return F.mse_loss(out, targets)
        else:
            return F.cross_entropy(out, targets)

    # --- 2. Gradient & HVP ---
    
    # Compute Gradient (1st Order)
    # grad(func) returns a function that takes the same args as func
    grad_fn = grad(loss_wrt_params)
    grads = grad_fn(params_flat)
    
    # Define HVP (2nd Order)
    # HVP is JVP of the Gradient function.
    def hvp(v):
        # JVP(f, primals, tangents)
        # Since loss_wrt_params only takes 1 arg (params_flat), 
        # we only pass 1 primal and 1 tangent.
        _, hv = jvp(grad_fn, (params_flat,), (v,))
        return hv

    # --- 3. Newton Solve for Lambda ---
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
            
        # Solve (H + lam*I) y = s (for derivative of lambda)
        y, mv_y = minres_solver(shifted_hvp, s, x_init=None, 
                                tol=config["minres_tol"], 
                                max_iter=config["minres_max_iter"])
        total_mv += mv_y
        
        dot_sy = torch.dot(s, y)
        F_prime = 1.0 + (M / 2.0) * (dot_sy / (norm_s + 1e-8))
        
        # Safe Newton update
        lam = max(0.0, float(lam - F_val / max(F_prime.item(), 1e-2)))
    
    # --- 4. Update ---
    if config["check_optimality"]:
        check_optimality(hvp, g, s, lam, M)
        
    new_params_flat = params_flat + s
    
    # Report Loss (using no_grad to be fast)
    with torch.no_grad():
        loss_val = loss_wrt_params(params_flat)
    
    return new_params_flat, loss_val, total_mv

# ==============================================================================
# 6. Main Execution
# ==============================================================================

def main():
    print(f"Running on: {CONFIG['device']}")
    torch.manual_seed(CONFIG['seed'])
    if CONFIG['device'] == 'cuda':
        torch.cuda.manual_seed(CONFIG['seed'])
        
    # 1. Prepare Data
    X, y, in_dim, out_dim, loss_type = prepare_data(
        CONFIG['dataset'], 
        CONFIG['num_samples'], 
        CONFIG['device'],
        dummy_dim=CONFIG['dummy_dim']
    )
    
    print(f"Data Shape: Input {X.shape}, Targets {y.shape}")
    print(f"Task Type: {loss_type.upper()}")
    
    # 2. Initialize Model
    model = nn.Sequential(
        nn.Linear(in_dim, CONFIG['hidden_dim']),
        nn.ReLU(),
        nn.Linear(CONFIG['hidden_dim'], CONFIG['hidden_dim']),
        nn.ReLU(),
        nn.Linear(CONFIG['hidden_dim'], CONFIG['hidden_dim']),
        nn.ReLU(),
        nn.Linear(CONFIG['hidden_dim'], out_dim)
    ).to(CONFIG['device'])
    
    print(f"Model Initialized: MLP {in_dim} -> {CONFIG['hidden_dim']} -> {CONFIG['hidden_dim']} -> {out_dim}")
    
    flat_params = torch.cat([p.view(-1) for p in model.parameters()])
    buffers = dict(model.named_buffers())
    
    print("-" * 65)
    print(f"{'Epoch':<5} | {'Loss':<12} | {'MatVecs':<8} | {'Time (s)'}")
    print("-" * 65)
    
    total_time = 0
    
    # 3. Optimization Loop
    for epoch in range(CONFIG['epochs']):
        start_t = time.time()
        
        new_flat_params, loss, mvs = cubic_regularized_step(
            model, flat_params, buffers, X, y, CONFIG, loss_type
        )
        
        flat_params = new_flat_params
        
        # Sync model (necessary if we want to check final accuracy using standard forward)
        idx = 0
        with torch.no_grad():
            for p in model.parameters():
                n = p.numel()
                p.copy_(flat_params[idx:idx+n].view(p.shape))
                idx += n
        
        end_t = time.time()
        dt = end_t - start_t
        total_time += dt
        
        print(f"{epoch:<5} | {loss.item():.6f}     | {mvs:<8} | {dt:.4f}")

    # 4. Final Accuracy Check
    if loss_type == 'cross_entropy':
        with torch.no_grad():
            logits = model(X)
            preds = torch.argmax(logits, dim=1)
            acc = (preds == y).float().mean()
            print("-" * 65)
            print(f"Final Training Accuracy: {acc.item()*100:.2f}%")

if __name__ == "__main__":
    main()