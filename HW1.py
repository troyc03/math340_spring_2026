import numpy as np

def fixed_point(f, x0, tol=1e-5, max_iter=1000):
    x_prev = x0
    for i in range(max_iter):
        x_curr = f(x_prev)
        if np.abs(x_curr - x_prev) < tol:
            return x_curr, i + 1
        x_prev = x_curr
    return x_curr, max_iter

def my_bisection_iterative(f, a, b, tol, max_steps=None):
    if np.sign(f(a)) == np.sign(f(b)):
        raise Exception("The scalars a and b do not bound a root.")
    
    step = 0
    while (b - a) / 2 > tol:
        step += 1
        m = (a + b) / 2
        if f(m) == 0:
            return m, 0
        if np.sign(f(a)) == np.sign(f(m)):
            a = m
        else:
            b = m
        if max_steps and step == max_steps:
            break
            
    approx_root = (a + b) / 2
    return approx_root, (b - a) / 2

# --- Exercise 3 Solutions ---

# i) Find p3 for f(x) = sqrt(x) - cos(x) on [0, 1] 
f3_i = lambda x: np.sqrt(x) - np.cos(x)
p3, _ = my_bisection_iterative(f3_i, 0, 1, tol=0, max_steps=3)
print(f"Exercise 3(i): p3 = {p3:.4f}")

# ii) Solutions for x^4 - 2x^3 - 4x^2 + 4x + 4 = 0 
f3_ii = lambda x: x**4 - 2*x**3 - 4*x**2 + 4*x + 4
root1, err1 = my_bisection_iterative(f3_ii, -2, -1, 1e-2)
root2, err2 = my_bisection_iterative(f3_ii, 2, 3, 1e-2)
print(f"Exercise 3(ii): Root 1 = {root1:.4f}, Root 2 = {root2:.4f}")

# --- Exercise 4 Solution ---

# Bisection for e^x - x^2 + 3x - 2 = 0 on [0, 1] 
f4 = lambda x: np.exp(x) - x**2 + 3*x - 2
root4, err4 = my_bisection_iterative(f4, 0, 1, 1e-5)
print(f"Exercise 4: Root = {root4:.6f} (Error bound: {err4:.6e})")

# --- Exercise 6 Solutions ---

# a) 2 + sin(x) - x = 0 => g(x) = 2 + sin(x) on [2, 3] 
g6_a = lambda x: 2 + np.sin(x)
root6_a, iters6_a = fixed_point(g6_a, 2.5, tol=1e-5)
print(f"Exercise 6(a): Root = {root6_a:.5f} in {iters6_a} iterations")

# b) x^3 - 2x - 5 = 0 => g(x) = (2x + 5)^(1/3) on [2, 3] 
g6_b = lambda x: (2*x + 5)**(1/3)
root6_b, iters6_b = fixed_point(g6_b, 2.5, tol=1e-5)
print(f"Exercise 6(b): Root = {root6_b:.5f} in {iters6_b} iterations")

# c) g(x) = 3x^2-e^x = 0 => x = (e^x / 3)^(1/2)
g6_c = lambda x: (np.exp(x) / 3)**0.5
root6_c, iters6_c = fixed_point(g6_c, 1.0, tol=1e-5)
print(f"Exercise 6(c): Root = {root6_c:.5f} in {iters6_c} iterations")

# d) x - cos(x) = 0 => g(x) = cos(x) 
g6_d = lambda x: np.cos(x)
root6_d, iters6_d = fixed_point(g6_d, 0.5, tol=1e-5)
print(f"Exercise 6(d): Root = {root6_d:.5f} in {iters6_d} iterations")