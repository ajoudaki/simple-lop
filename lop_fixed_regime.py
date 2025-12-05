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
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# START: CORRECTED Muon Optimizer Implementation
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class Muon(Optimizer):
    """
    Implements the SCALABLE Muon optimizer.
    """
    def __init__(self, params, lr=1e-3,
                 mu=0.95, k=5, rms_match_scale=0.2, weight_decay=0.1,
                 adam_betas=(0.9, 0.999), adam_eps=1e-8):
        
        if not 0.0 <= lr: raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= mu < 1.0: raise ValueError(f"Invalid mu: {mu}")
        if not 1 <= k: raise ValueError(f"Invalid k: {k}")
        
        self.a, self.b, self.c = 3.4445, -4.7750, 2.0315
            
        defaults = dict(lr=lr, mu=mu, k=k, rms_match_scale=rms_match_scale,
                        weight_decay=weight_decay, adam_betas=adam_betas, adam_eps=adam_eps)
        super().__init__(params, defaults)

    def _newton_schulz_ortho(self, M, K):
        M_norm = torch.norm(M, 'fro') + 1e-12
        X = M / M_norm
        for _ in range(K):
            X_XT = X @ X.T
            X = self.a * X + self.b * (X_XT @ X) + self.c * (X_XT @ X_XT @ X)
        return X

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            wd = group['weight_decay']
            adam_beta1, adam_beta2 = group['adam_betas']
            adam_eps = group['adam_eps']
            mu, k, rms_scale = group['mu'], group['k'], group['rms_match_scale']

            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                if grad.is_sparse: raise RuntimeError('Muon does not support sparse grads.')
                state = self.state[p]

                # Muon for 2D weights
                if p.dim() == 2:
                    m, n = p.shape
                    if 'momentum' not in state: state['momentum'] = torch.zeros_like(p)
                    M = state['momentum']
                    M.mul_(mu).add_(grad)
                    O_t = self._newton_schulz_ortho(M, k)
                    dim_scale = math.sqrt(max(m, n))
                    if wd != 0: p.add_(p, alpha=-lr * wd)
                    p.add_((rms_scale * dim_scale) * O_t, alpha=-lr)
                # AdamW for vectors/biases
                else:
                    if 'step' not in state:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)
                    state['step'] += 1
                    step = state['step']
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    if wd != 0: p.add_(p, alpha=-lr * wd)
                    exp_avg.mul_(adam_beta1).add_(grad, alpha=1.0 - adam_beta1)
                    exp_avg_sq.mul_(adam_beta2).addcmul_(grad, grad, value=1.0 - adam_beta2)
                    denom = (exp_avg_sq.sqrt() / math.sqrt(1.0 - adam_beta2 ** step)).add_(adam_eps)
                    p.addcdiv_(exp_avg, denom, value=-lr / (1.0 - adam_beta1 ** step))
        return loss

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# METRICS
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def effective_rank(Z, eps=1e-10):
    # Z should be (N_samples, D_features)
    Z_c = Z - Z.mean(dim=0, keepdim=True)
    C = (Z_c.T @ Z_c) / (max(1, Z.shape[0] - 1))
    C = C + eps * torch.eye(C.shape[0], device=C.device, dtype=C.dtype)
    try:
        eigvals = torch.linalg.eigvalsh(C)
    except:
        return torch.tensor(1.0, device=Z.device, dtype=Z.dtype)
    eigvals = torch.clamp(eigvals, min=eps)
    norm_eigvals = eigvals / eigvals.sum()
    entropy = -torch.sum(norm_eigvals * torch.log(norm_eigvals))
    return torch.exp(entropy)

def fraction_duplicate_features(A, threshold=0.95):
    # A should be (N_samples, D_features)
    hidden_dim = A.shape[1]
    if hidden_dim == 0: return torch.tensor(0.0, device=A.device)
    corr_matrix = torch.corrcoef(A.T)
    corr_matrix = torch.nan_to_num(corr_matrix, nan=0.0)
    upper_tri_abs = torch.abs(torch.triu(corr_matrix, diagonal=1))
    is_duplicate = torch.any(upper_tri_abs > threshold, dim=0)
    return torch.sum(is_duplicate) / hidden_dim

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# MODEL DEFINITIONS
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class BaseModel(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.activations_dict = {
            'relu': nn.ReLU(),
            'selu': nn.SELU(),
            'gelu': nn.GELU(),
            'erf': lambda x: torch.erf(x),
            'tanh': nn.Tanh()
        }
        if activation.lower() not in self.activations_dict:
            raise ValueError(f"Unknown activation: {activation}")
        self.activation_fn = self.activations_dict[activation.lower()]
        self.activation_name = activation.lower()

    def activation_derivative(self, z):
        derivatives = {
            'relu': lambda x: (x > 0).double(),
            'selu': lambda x: torch.where(x > 0, torch.tensor(1.0507, device=x.device, dtype=x.dtype),
                                         1.0507 * 1.67326 * torch.exp(x)),
            'gelu': lambda x: torch.sigmoid(1.702 * x) * (1 + 1.702 * x * torch.sigmoid(1.702 * x) * (1 - torch.sigmoid(1.702 * x))),
            'erf': lambda x: (2.0 / math.sqrt(math.pi)) * torch.exp(-x**2),
            'tanh': lambda x: 1 - torch.tanh(x)**2
        }
        return torch.abs(derivatives[self.activation_name](z))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                init.xavier_normal_(m.weight)
                if m.bias is not None: init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                if m.weight is not None: init.constant_(m.weight, 1)
                if m.bias is not None: init.constant_(m.bias, 0)

class MLP(BaseModel):
    def __init__(self, d=2, h=100, L=5, activation='tanh', use_bn=False, use_ln=False):
        super().__init__(activation)
        self.L = L
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        self.layers.append(nn.Linear(d, h))
        self.norms.append(self._get_norm(h, use_bn, use_ln))
        
        for _ in range(L - 1):
            self.layers.append(nn.Linear(h, h))
            self.norms.append(self._get_norm(h, use_bn, use_ln))
            
        self.output_layer = nn.Linear(h, 1)
        self._init_weights()

    def _get_norm(self, dim, use_bn, use_ln):
        if use_bn: return nn.BatchNorm1d(dim)
        if use_ln: return nn.LayerNorm(dim)
        return nn.Identity()

    def forward(self, X, capture=False):
        Zs, As = [], []
        A = X
        for layer, norm in zip(self.layers, self.norms):
            Z = layer(A)
            Z = norm(Z) 
            A = self.activation_fn(Z)
            if capture:
                Zs.append(Z)
                As.append(A)
        logits = self.output_layer(A)
        p = torch.sigmoid(logits)
        if capture: return p, logits, Zs, As
        return p, logits

class CNN(BaseModel):
    """
    Standard 1D CNN. 
    Metric flattening: (B, C, L) -> (B*L, C) so Channel is the feature.
    """
    def __init__(self, d=2, h=100, L=5, activation='tanh', use_bn=False, use_ln=False):
        super().__init__(activation)
        self.L = L
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        self.convs.append(nn.Conv1d(1, h, kernel_size=3, padding=1))
        self.norms.append(self._get_norm(h, use_bn, use_ln))
        
        for _ in range(L - 1):
            self.convs.append(nn.Conv1d(h, h, kernel_size=3, padding=1))
            self.norms.append(self._get_norm(h, use_bn, use_ln))
            
        self.fc = nn.Linear(h * d, 1)
        self._init_weights()

    def _get_norm(self, dim, use_bn, use_ln):
        if use_bn: return nn.BatchNorm1d(dim)
        if use_ln: return nn.GroupNorm(1, dim) 
        return nn.Identity()

    def forward(self, X, capture=False):
        A = X.unsqueeze(1)
        Zs, As = [], []
        
        for conv, norm in zip(self.convs, self.norms):
            Z = conv(A)
            Z = norm(Z)
            A = self.activation_fn(Z)
            
            if capture:
                # Flatten Batch*Length, Channel
                Zs.append(Z.permute(0, 2, 1).reshape(-1, Z.shape[1]))
                As.append(A.permute(0, 2, 1).reshape(-1, A.shape[1]))
                
        A_flat = A.flatten(1)
        logits = self.fc(A_flat)
        p = torch.sigmoid(logits)
        if capture: return p, logits, Zs, As
        return p, logits

class ResNet(BaseModel):
    """
    Standard ResNet.
    Metric flattening: (B, C, L) -> (B*L, C).
    """
    class ResBlock(nn.Module):
        def __init__(self, h, activation_fn, norm_builder):
            super().__init__()
            self.act = activation_fn
            self.conv1 = nn.Conv1d(h, h, kernel_size=3, padding=1)
            self.norm1 = norm_builder(h)
            self.conv2 = nn.Conv1d(h, h, kernel_size=3, padding=1)
            self.norm2 = norm_builder(h)
        
        def forward(self, x, capture_list=None):
            residual = x
            z1 = self.conv1(x)
            z1 = self.norm1(z1)
            a1 = self.act(z1)
            z2 = self.conv2(a1)
            z2 = self.norm2(z2)
            z_out = residual + z2
            a_out = self.act(z_out)
            
            if capture_list is not None:
                C = z_out.shape[1]
                capture_list[0].append(z_out.permute(0, 2, 1).reshape(-1, C))
                capture_list[1].append(a_out.permute(0, 2, 1).reshape(-1, C))
                
            return a_out

    def __init__(self, d=2, h=100, L=5, activation='tanh', use_bn=False, use_ln=False):
        super().__init__(activation)
        self.L = L
        self.input_proj = nn.Linear(d, h) 
        self.blocks = nn.ModuleList()
        norm_builder = lambda dim: self._get_norm(dim, use_bn, use_ln)
        
        for _ in range(L):
            self.blocks.append(self.ResBlock(h, self.activation_fn, norm_builder))
            
        self.output_layer = nn.Linear(h, 1)
        self._init_weights()

    def _get_norm(self, dim, use_bn, use_ln):
        if use_bn: return nn.BatchNorm1d(dim)
        if use_ln: return nn.GroupNorm(1, dim)
        return nn.Identity()

    def forward(self, X, capture=False):
        x_embed = self.input_proj(X)
        A = x_embed.unsqueeze(2) # (B, h, 1)
        Zs, As = [], []
        capture_list = (Zs, As) if capture else None
        
        for block in self.blocks:
            A = block(A, capture_list)
            
        A_pool = A.mean(dim=2)
        logits = self.output_layer(A_pool)
        p = torch.sigmoid(logits)
        if capture: return p, logits, Zs, As
        return p, logits

class ViT(BaseModel):
    """
    Standard ViT.
    Metric flattening: (B, L, D) -> (B*L, D).
    """
    def __init__(self, d=2, h=100, L=5, activation='tanh', use_bn=False, use_ln=False):
        super().__init__(activation)
        self.L = L
        self.seq_len = 4 
        self.embed_dim = h
        self.embedding = nn.Linear(d, self.seq_len * self.embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, self.embed_dim))
        
        self.blocks = nn.ModuleList()
        for _ in range(L):
            self.blocks.append(self.ViTBlock(self.embed_dim, self.activation_fn, use_bn, use_ln))
            
        self.norm = self._get_norm(self.embed_dim, use_bn, use_ln)
        self.head = nn.Linear(self.embed_dim, 1) 
        self._init_weights()

    def _get_norm(self, dim, use_bn, use_ln):
        if use_bn: return nn.BatchNorm1d(dim)
        if use_ln: return nn.LayerNorm(dim)
        return nn.Identity()

    class ViTBlock(nn.Module):
        def __init__(self, dim, act, use_bn, use_ln):
            super().__init__()
            self.act = act
            self.use_bn = use_bn
            self.norm1 = self._make_norm(dim, use_bn, use_ln)
            self.norm2 = self._make_norm(dim, use_bn, use_ln)
            self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)
            self.linear1 = nn.Linear(dim, dim * 4)
            self.linear2 = nn.Linear(dim * 4, dim)

        def _make_norm(self, dim, use_bn, use_ln):
            if use_bn: return nn.BatchNorm1d(dim)
            if use_ln: return nn.LayerNorm(dim)
            return nn.Identity()
            
        def _apply_norm(self, x, norm_layer):
            if isinstance(norm_layer, nn.BatchNorm1d):
                return norm_layer(x.transpose(1, 2)).transpose(1, 2)
            return norm_layer(x)

        def forward(self, x, capture_list=None):
            resid = x
            n1 = self._apply_norm(x, self.norm1)
            attn_out, _ = self.attn(n1, n1, n1)
            x = resid + attn_out
            
            resid = x
            n2 = self._apply_norm(x, self.norm2)
            z_mlp = self.linear1(n2)
            a_mlp = self.act(z_mlp)
            x = resid + self.linear2(a_mlp)
            
            if capture_list:
                # Shape (B, Seq, Dim_hidden) -> (B*Seq, Dim_hidden)
                D = z_mlp.shape[-1]
                capture_list[0].append(z_mlp.reshape(-1, D))
                capture_list[1].append(a_mlp.reshape(-1, D))
                
            return x

    def forward(self, X, capture=False):
        Zs, As = [], []
        capture_list = (Zs, As) if capture else None
        B = X.shape[0]
        x = self.embedding(X) 
        x = x.view(B, self.seq_len, self.embed_dim)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x, capture_list)
            
        if isinstance(self.norm, nn.BatchNorm1d):
            x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        else:
            x = self.norm(x)
            
        x_pool = x.mean(dim=1)
        logits = self.head(x_pool)
        p = torch.sigmoid(logits)
        if capture: return p, logits, Zs, As
        return p, logits

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# EXPERIMENT LOOP
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def make_gmm_task(rng, n=96, d=2, k=4, delta=3.0, spread=0.8, sigma=0.22, relabel_p=0.0):
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
    return X[perm].astype(np.float64), y[perm].astype(np.float64)

def get_optimizer(model_params, args):
    opt_name = args.optimizer.lower()
    if opt_name == 'sgd':
        return optim.SGD(model_params, lr=args.lr, weight_decay=args.weight_decay)
    elif opt_name == 'adam':
        return optim.AdamW(model_params, lr=args.lr,
                           betas=args.adam_betas, eps=args.adam_eps,
                           weight_decay=args.weight_decay)
    elif opt_name == 'muon':
        return Muon(model_params, lr=args.lr,
                    mu=args.muon_mu, k=args.muon_k, rms_match_scale=args.muon_rms_scale,
                    weight_decay=args.weight_decay,
                    adam_betas=args.adam_betas, adam_eps=args.adam_eps)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

def get_model(args):
    common_kwargs = {
        'd': args.d, 'h': args.h, 'L': args.L, 
        'activation': args.activation,
        'use_bn': args.use_bn, 'use_ln': args.use_ln
    }
    if args.model.lower() == 'mlp': return MLP(**common_kwargs)
    elif args.model.lower() == 'cnn': return CNN(**common_kwargs)
    elif args.model.lower() == 'resnet': return ResNet(**common_kwargs)
    elif args.model.lower() == 'vit': return ViT(**common_kwargs)
    else: raise ValueError(f"Unknown model architecture: {args.model}")

def run(args):
    print(f"Device: {device} | Model: {args.model} | Opt: {args.optimizer}")
    print(f"Conf: BN={args.use_bn} LN={args.use_ln} Act={args.activation}")
    
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    net = get_model(args).to(device).double()
    optimizer = get_optimizer(net.parameters(), args)
    criterion = nn.BCEWithLogitsLoss()

    steps, n = args.steps, args.n
    fresh_steps = args.fresh_steps if args.fresh_steps is not None else args.steps

    min_losses, final_losses, gnorms = [], [], []
    eff_rank_data = {f'eff_rank_L{i}': [] for i in range(net.L)}
    dup_frac_data = {f'dup_frac_L{i}': [] for i in range(net.L)}
    frozen_frac_data = {f'frozen_frac_L{i}': [] for i in range(net.L)}

    print(f"Running {args.tasks} tasks...")
    
    for t in range(args.tasks):
        X_np, y_np = make_gmm_task(rng, n=n, d=args.d, k=args.k,
                                   delta=args.delta, spread=args.spread, 
                                   sigma=args.sigma, relabel_p=args.relabel)
        X = torch.from_numpy(X_np).to(device).double()
        y = torch.from_numpy(y_np).to(device).double()
        
        min_ce, final_ce, g0 = 1e9, 1e9, None
        
        for s in range(steps):
            optimizer.zero_grad()
            p, logits, Zs, As = net(X, capture=True)
            loss = criterion(logits, y)
            ce = loss.item()
            min_ce = min(min_ce, ce)
            if s == steps - 1: final_ce = ce
            loss.backward()
            if s == 0:
                g0 = math.sqrt(sum(p.grad.norm(2).item()**2 
                                   for p in net.parameters() 
                                   if p.grad is not None and p.requires_grad))
            optimizer.step()
        
        # Metrics Calculation
        with torch.no_grad():
            _, _, Zs_final, As_final = net(X, capture=True)

        for i, Z in enumerate(Zs_final):
            is_frozen = net.activation_derivative(Z) < args.frozen_thresh 
            frozen_prob = torch.mean(is_frozen.double(), dim=0)
            is_frozen_unit = frozen_prob > args.frozen_p_thresh
            frozen_frac_data[f'frozen_frac_L{i}'].append(torch.mean(is_frozen_unit.double()).item())
            eff_rank_data[f'eff_rank_L{i}'].append(effective_rank(Z).item())

        for i, A in enumerate(As_final):
            dup_frac_data[f'dup_frac_L{i}'].append(fraction_duplicate_features(A, args.corr_thresh).item())

        gnorms.append(g0 if g0 is not None else 0.0)
        min_losses.append(min_ce)
        final_losses.append(final_ce)

        if (t + 1) % (max(1, args.tasks // 10)) == 0:
            print(f"  Task {t+1}/{args.tasks} completed.")


    # --- Fresh Baseline Calculation ---
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
            
            # Create a fresh model and optimizer of the SAME selected type
            f_net = get_model(args).to(device).double()
            f_opt = get_optimizer(f_net.parameters(), args)
            
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


    # Results gathering
    all_task_results = {
        "mean_min_CE": float(np.mean(min_losses)),
        "mean_final_CE": float(np.mean(final_losses)),
        "median_min_CE": float(np.median(min_losses)),
        "median_final_CE": float(np.median(final_losses)),
        "mean_grad_norm": float(np.mean(gnorms)),
        "fresh_mean_final_CE": fresh_mean,
        "fresh_median_final_CE": fresh_median,
    }
    for i in range(net.L):
        all_task_results[f'mean_frozen_L{i}'] = float(np.mean(frozen_frac_data[f'frozen_frac_L{i}']))
        all_task_results[f'mean_eff_rank_L{i}'] = float(np.mean(eff_rank_data[f'eff_rank_L{i}']))
        all_task_results[f'mean_dup_L{i}'] = float(np.mean(dup_frac_data[f'dup_frac_L{i}']))
    
    print("\n--- Results Summary ---")
    print(json.dumps(all_task_results, indent=2))

    # Save CSV
    df_data = {
        "task": np.arange(1, args.tasks + 1),
        "min_CE": min_losses,
        "final_CE": final_losses,
        "grad_norm": gnorms
    }
    for i in range(net.L):
        df_data[f'frozen_frac_L{i}'] = frozen_frac_data[f'frozen_frac_L{i}']
        df_data[f'eff_rank_L{i}'] = eff_rank_data[f'eff_rank_L{i}']
        df_data[f'dup_frac_L{i}'] = dup_frac_data[f'dup_frac_L{i}']
        
    df = pd.DataFrame(df_data)
    df.to_csv("out/lop_results.csv", index=False)
    
    # Save Plots
    import os
    os.makedirs("out", exist_ok=True)
    plt.style.use('ggplot')
    
    # 1. Min CE
    plt.figure(figsize=(10, 6))
    plt.plot(df["task"], df["min_CE"])
    plt.xlabel("Task")
    plt.ylabel("Min CE")
    plt.title(f"Min CE ({args.model})")
    plt.savefig("out/lop_minCE.png", dpi=120)
    plt.close()

    # 2. Frozen Fraction
    plt.figure(figsize=(10, 6))
    colors = plt.cm.coolwarm(np.linspace(0, 1, net.L))
    for i in range(net.L):
        plt.plot(df["task"], df[f'frozen_frac_L{i}'], label=f'L{i}', color=colors[i], alpha=0.8)
    plt.xlabel("Task")
    plt.ylabel("Frozen Fraction")
    plt.legend()
    plt.title("Frozen Fraction")
    plt.savefig("out/lop_frozen.png", dpi=120)
    plt.close()
    
    # 3. Effective Rank
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, net.L))
    for i in range(net.L):
        plt.plot(df["task"], df[f'eff_rank_L{i}'], label=f'L{i}', color=colors[i], alpha=0.8)
    plt.xlabel("Task")
    plt.ylabel("Effective Rank")
    plt.legend()
    plt.title("Effective Rank")
    plt.savefig("out/lop_effrank.png", dpi=120)
    plt.close()

    # 4. Final CE
    plt.figure(figsize=(10, 6))
    plt.plot(df["task"], df["final_CE"])
    plt.xlabel("Task")
    plt.ylabel("Final CE")
    plt.title(f"Final CE ({args.model})")
    plt.savefig("out/lop_finalCE.png", dpi=120)
    plt.close()

    # 5. Gradient Norm
    plt.figure(figsize=(10, 6))
    plt.plot(df["task"], df["grad_norm"])
    plt.xlabel("Task")
    plt.ylabel("Gradient Norm")
    plt.title(f"Gradient Norm (Step 0) ({args.model})")
    plt.savefig("out/lop_grad.png", dpi=120)
    plt.close()

    # 6. Duplicate Fraction
    plt.figure(figsize=(10, 6))
    colors = plt.cm.plasma(np.linspace(0, 1, net.L))
    for i in range(net.L):
        plt.plot(df["task"], df[f'dup_frac_L{i}'], label=f'L{i}', color=colors[i], alpha=0.8)
    plt.xlabel("Task")
    plt.ylabel("Duplicate Fraction")
    plt.legend()
    plt.title("Duplicate Feature Fraction")
    plt.savefig("out/lop_dup_frac.png", dpi=120)
    plt.close()
    
    print("Results and all 6 plots saved to out/")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="MLP", choices=["MLP", "CNN", "ResNet", "ViT"])
    ap.add_argument("--use_bn", action='store_true')
    ap.add_argument("--use_ln", action='store_true')
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--d", type=int, default=2)
    ap.add_argument("--h", type=int, default=100)
    ap.add_argument("--L", type=int, default=5)
    ap.add_argument("--activation", type=str, default="tanh", choices=["relu", "selu", "gelu", "erf", "tanh"])
    ap.add_argument("--frozen_thresh", type=float, default=0.05)
    ap.add_argument("--frozen_p_thresh", type=float, default=0.98)
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.6)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--fresh_steps", type=int, default=None)
    ap.add_argument("--tasks", type=int, default=100)
    ap.add_argument("--delta", type=float, default=0.0)
    ap.add_argument("--spread", type=float, default=1.0)
    ap.add_argument("--sigma", type=float, default=0.2)
    ap.add_argument("--relabel", type=float, default=0.0)
    ap.add_argument("--corr_thresh", type=float, default=0.95)
    ap.add_argument("--fresh_baseline", type=int, default=10)
    
    ap.add_argument("--optimizer", type=str, default="SGD", choices=["SGD", "Adam", "Muon"])
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--muon_mu", type=float, default=0.95)
    ap.add_argument("--muon_k", type=int, default=5)
    ap.add_argument("--muon_rms_scale", type=float, default=0.2)
    ap.add_argument("--adam_betas", type=float, nargs=2, default=[0.9, 0.999])
    ap.add_argument("--adam_eps", type=float, default=1e-8)

    args = ap.parse_args()
    import os
    os.makedirs("out", exist_ok=True)
    run(args)