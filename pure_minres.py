import torch

# --- 1. The Matrix Operator (Tracks MatVecs) ---
class CubicNewtonOperator:
    def __init__(self, H):
        self.H = H
        self.dim = H.shape[0]
        self.matvec_count = 0
        self.current_lambda = 0.0
        
    def set_lambda(self, lam):
        self.current_lambda = lam
        
    def __call__(self, v):
        self.matvec_count += 1
        # Returns (H + lambda*I) v
        return torch.mv(self.H, v) + self.current_lambda * v

# --- 2. Standard MINRES Implementation ---
def minres(A_op, b, x_init=None, tol=1e-5, max_iter=1000):
    """
    MINRES solver with Warm Start capability.
    If x_init is provided, we solve for the correction: A * delta = b - A * x_init
    """
    device = b.device
    
    # Handle initialization
    if x_init is not None:
        x = x_init.clone()
        # Calculate initial residual: r0 = b - A @ x_init
        # NOTE: This consumes 1 MatVec
        r0 = b - A_op(x)
    else:
        x = torch.zeros_like(b)
        r0 = b.clone()
    
    # Standard MINRES setup using r0 as the starting vector
    v_old = torch.zeros_like(b)
    v = r0.clone()
    beta = torch.norm(v)
    
    # Early exit if residual is already small (Good guess!)
    if beta < 1e-15:
        return x
        
    v = v / beta
    
    # Variables for the correction vector 'd' (in standard notation x = x + alpha * d)
    # We are essentially accumulating corrections onto our initial 'x'
    
    c_old, s_old = 1.0, 0.0
    c, s = 1.0, 0.0
    
    w_old = torch.zeros_like(b)
    w = torch.zeros_like(b)
    
    beta_1 = beta.item()
    norm_b = torch.norm(b) # Relative error is typically against original b, not residual
    if norm_b < 1e-15: norm_b = 1.0
    
    for k in range(max_iter):
        # Lanczos Step
        Av = A_op(v)
        alpha = torch.dot(v, Av)
        v_next = Av - alpha * v - beta * v_old
        beta_next = torch.norm(v_next)
        
        # QR / Givens Rotation
        delta = c * alpha + s * beta
        gamma = s * alpha - c * beta
        epsilon = beta_next
        
        r_1 = torch.sqrt(delta**2 + epsilon**2)
        
        if r_1 < 1e-15: c_new, s_new = 1.0, 0.0
        else: c_new, s_new = delta/r_1, epsilon/r_1
            
        # Update solution x
        eta = c_new * beta_1
        d = (v - gamma * w - s_old * beta * w_old) / r_1
        x = x + eta * d
        
        # Update Residual Norm
        beta_1 = -s_new * beta_1
        
        # Shift Variables
        v_old = v.clone()
        v = v_next / beta_next
        beta = beta_next
        w_old = w.clone()
        w = d.clone()
        c_old, s_old = c, s
        c, s = c_new, s_new
        
        # Check Convergence
        if abs(beta_1) / norm_b < tol:
            break
            
    return x


def solve_cubic_subproblem(H, g, M, tol=1e-2, max_newton_iter=20):
    n = H.shape[0]
    op = CubicNewtonOperator(H)
    
    lam = 0.0 
    
    # Initialize guess for s
    s_guess = torch.zeros_like(g)
    
    print(f"{'Iter':<5} | {'Lambda':<10} | {'||s||':<10} | {'Error F(lam)':<12} | {'MatVecs'}")
    print("-" * 65)
    
    for k in range(max_newton_iter):
        op.set_lambda(lam)
        
        # --- MINRES 1: Solve s ---
        # Warm start using the s from the previous iteration
        s = minres(op, -g, x_init=s_guess, tol=1e-2)
        
        # Update the guess for the next loop
        s_guess = s.clone()
        
        norm_s = torch.norm(s)
        F_val = lam - (M / 2.0) * norm_s
        
        print(f"{k:<5} | {lam:.4f}     | {norm_s:.4f}     | {abs(F_val):.2e}    | {op.matvec_count}")
        
        if abs(F_val) < tol and k > 0:
            print("Newton Converged.")
            break
            
        # --- MINRES 2: Solve y ---
        # We can technically warm start y too if we stored it, 
        # but y changes direction more than s usually. 
        # For simplicity, we leave y cold-started here.
        y = minres(op, s, tol=1e-5)
        
        # --- Newton Update ---
        s_dot_y = torch.dot(s, y)
        derivative_term = (M / 2.0) * (s_dot_y / norm_s)
        F_prime = 1.0 + derivative_term
        
        if F_prime < 0.1: F_prime = 1.0
        
        lam_new = lam - (F_val / F_prime)
        
        if isinstance(lam_new, torch.Tensor): lam_new = lam_new.item()
        
        if lam_new < 0: lam = 0.0
        else: lam = lam_new

    return s, lam, op.matvec_count

# --- 4. Verification & Experiment ---
def run_check():
    torch.manual_seed(123)
    N = 1000
    M = 10.0  # Cubic penalty strength
    
    # 1. Create Problem (Indefinite Hessian)
    # We create eigenvalues explicitly to verify optimality later
    # 20 negative eigenvalues, 80 positive
    eigs = torch.linspace(-5, 10, N) 
    Q, _ = torch.linalg.qr(torch.randn(N, N))
    H = Q @ torch.diag(eigs) @ Q.T
    
    # Random gradient
    g = torch.randn(N)
    
    print("Problem Setup:")
    print(f"Hessian Eigenvalues: Min {eigs.min():.2f}, Max {eigs.max():.2f}")
    print(f"Cubic M: {M}")
    print("Starting Solver...\n")
    
    # 2. Solve
    s_star, lam_star, total_matvecs = solve_cubic_subproblem(H, g, M)
    
    print("\n" + "="*30)
    print("OPTIMALITY CONDITIONS CHECK")
    print("="*30)
    
    # --- Condition 1: Stationarity ---
    # (H + lambda*I)s + g = 0
    residual_vec = (H @ s_star + lam_star * s_star) + g
    resid_norm = torch.norm(residual_vec)
    print(f"1. Stationarity Error ||(H+lamI)s + g||: {resid_norm:.2e}")
    
    # --- Condition 2: Norm Matching ---
    # lambda = (M/2)||s||
    norm_s = torch.norm(s_star)
    expected_lam = (M / 2.0) * norm_s
    lam_diff = abs(lam_star - expected_lam)
    print(f"2. Norm Condition Error |lam - M/2||s||| : {lam_diff:.2e}")
    print(f"   (Lambda: {lam_star:.4f}, Expected: {expected_lam:.4f})")
    
    # --- Condition 3: PSD Condition ---
    # H + lambda*I >= 0 (shifted hessian must be PSD)
    # We check this by computing eigenvalues of the shift
    shifted_eigs = torch.linalg.eigvalsh(H + lam_star * torch.eye(N))
    min_shifted_eig = shifted_eigs.min()
    
    is_psd = min_shifted_eig >= -1e-4 # Allow tiny numerical error
    status = "[PASS]" if is_psd else "[FAIL]"
    
    print(f"3. PSD Condition (H + lamI >= 0)       : {status}")
    print(f"   Min Eigenvalue of (H + lamI)        : {min_shifted_eig:.4e}")
    
    # --- Efficiency ---
    print("-" * 30)
    print(f"Total Matrix-Vector Products Used: {total_matvecs}")

if __name__ == "__main__":
    run_check()