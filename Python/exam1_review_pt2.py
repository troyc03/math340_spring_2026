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

# Calculation
lambda_root = newton_method(0.1)
print(f"Approximation for lambda: {lambda_root:.5f}")

# Plot results
lam_values = np.linspace(0.01, 0.5, 100)
f_values = [f(lam) for lam in lam_values]
plt.plot(lam_values, f_values, label='f(lambda)')
plt.axhline(0, color='red', linestyle='--', label='Zero Line')
plt.scatter(lambda_root, 0, color='green', label=f'Solution: {lambda_root:.5f}')
plt.axvline(x=lambda_root, color='blue', linestyle='--', label=f'lambda={lambda_root:.5f}')
plt.scatter([0.1], [f(0.1)], color='orange', label='Initial Guess')
plt.title('Function f(lambda) for Exercise 5')
plt.xlabel('lambda')
plt.ylabel('f(lambda)')
plt.legend()
plt.grid(True)
plt.show()