# =======================
# Exam 1 Review - Part 2
# =======================

# ==============================
# --- Exercise 4 Solution ---
# ==============================

import math

# Census Data
census = {
    1950: 151326,
    1960: 179323,
    1970: 203302,
}

P0 = census[1950]
P1 = census[1960]
P2 = census[1970]

# Algebraic formula
PL_target = (2 * P0 * P1 * P2 - P1**2 * (P0 + P2)) / (P0 * P2 - P1**2)
print(f"Algebraic target PL: {PL_target}")

def f(PL):
    if PL == P0: return 1e18 # avoid div by zero
    numerator_u = (PL / P1) - 1
    denominator_u = (PL / P0) - 1
    u = numerator_u / denominator_u
    prediction_P2 = PL / (1 + (denominator_u * (u**2)))
    return prediction_P2 - P2

# Check derivative at PL_target
h = 1.0
df = (f(PL_target + h) - f(PL_target)) / h
print(f"f(PL_target): {f(PL_target):.5f}")
print(f"df at PL_target: {df:.5f}")

# Let's try to run Newton's method with a slightly better h or analytical derivative
def df_numeric(p, h=1.0):
    return (f(p + h) - f(p)) / h

def newton(guess, tol=1e-5, max_iter=20):
    p = guess
    for i in range(max_iter):
        val = f(p)
        der = df_numeric(p)
        if der == 0:
            print(f"Zero derivative at p={p:.5f}")
            break
        p_next = p - val / der
        print(f"Iter {i}: p={p:.5f}, f(p)={val:.5f}, df={der:.5f}, p_next={p_next:.5f}")
        if abs(p_next - p) < tol:
            return p_next
        p = p_next
    return p

print("\nRunning Newton with guess 450000:")
res = newton(450000)
print(f"Result: {res:.5f}")

# Predictions for 1980 (t=30) and 2010 (t=60)
PL = res
c = (PL / P0) - 1
u = ((PL / P1) - 1) / ((PL / P0) - 1)
k = -math.log(u) / 10

def P(t):
    return PL / (1 + c * math.exp(-k * t))

print(f"c: {c:.5f}, k: {k:.5f}")
print(f"P(30) (1980): {P(30):.5f}")
print(f"P(60) (2010): {P(60):.5f}")
print('-' * 50)

# Plot results
import numpy as np
import matplotlib.pyplot as plt

t_values = np.linspace(0, 60, 100)
P_values = [P(t) for t in t_values]
plt.plot(t_values, P_values, label='Predicted Population')
plt.axhline(y=P0, color='gray', linestyle='--', label='1950')
plt.axhline(y=P1, color='blue', linestyle='--', label='1960')
plt.axhline(y=P2, color='green', linestyle='--', label='1970')
plt.axvline(x=30, color='orange', linestyle='--', label='1980')
plt.axvline(x=60, color='red', linestyle='--', label='2010')
plt.scatter([0, 10, 20], [P0, P1, P2], color='red', label='Census Data')
plt.scatter([30, 60], [P(30), P(60)], color='purple', label='Predictions') 
plt.xlabel('Years since 1950')
plt.ylabel('Population')
plt.title('Population Growth Prediction')
plt.legend()
plt.grid(True)
plt.show()

# ===========================
# --- Exercise 5 Solution ---
# ===========================

# Constants from Problem 16
N0 = 1_000_000
N1 = 1_564_000
v = 435_000

def f(lam):
    # Standard solution equation minus the target N(1)
    return N0 * math.exp(lam) + (v / lam) * (math.exp(lam) - 1) - N1

def df(lam):
    # Derivative of f(lam) with respect to lam
    exp_lam = math.exp(lam)
    term1 = N0 * exp_lam
    term2 = v * (lam * exp_lam - (exp_lam - 1)) / (lam**2)
    return term1 + term2

def newton_method(guess, tol=1e-5):
    p_n = guess
    for i in range(100):
        f_val = f(p_n)
        df_val = df(p_n)
        p_next = p_n - f_val / df_val
        if abs(p_next - p_n) < tol:
            return p_next
        p_n = p_next
    return p_n

# Secant method for comparison
def secant_method(x0, x1, tol=1e-5):
    for i in range(100):
        f_x0 = f(x0)
        f_x1 = f(x1)
        if f_x1 - f_x0 == 0:
            print("Zero difference in function values, cannot continue.")
            return None
        x2 = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
        if abs(x2 - x1) < tol:
            return x2
        x0, x1 = x1, x2
    return x1

# Calculation
lambda_root = newton_method(0.1)
lambda_secant = secant_method(0.1, 0.2)
print(f"Approximation for lambda: {lambda_root:.5f}")
print(f"Secant method approximation for lambda: {lambda_secant:.5f}")

# Plot results
lam_values = np.linspace(0.01, 0.5, 100)
f_values = [f(lam) for lam in lam_values]
plt.subplots(figsize=(10, 6))
plt.plot(lam_values, f_values, label='f(lambda)')
plt.axhline(0, color='red', linestyle='--', label='Zero Line')
plt.axhline(0, color='red', linestyle='--')
plt.axvline(lambda_root, color='green', linestyle='--', label=f'Newton lambda: {lambda_root:.5f}')
plt.axvline(lambda_secant, color='orange', linestyle='--', label=f'Secant lambda: {lambda_secant:.5f}')
plt.scatter(lambda_root, f(lambda_root), color='green', label=f'Newton Solution: {lambda_root:.5f}')
plt.scatter(lambda_secant, f(lambda_secant), color='orange', label=f'Secant Solution: {lambda_secant:.5f}')
plt.xlabel('lambda')
plt.ylabel('f(lambda)')
plt.title('Finding lambda for Exercise 5')
plt.legend()
plt.grid(True)
plt.show()