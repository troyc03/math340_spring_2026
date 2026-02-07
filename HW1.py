# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 11:42:19 2026

@author: Troy
"""

# Exercise 5

import numpy as np

def fixed_point(f, x0, tol=1e-5, max_iter=1000):
    # Previous guess
    x_prev = x0
    # Iterate over maximum iterations
    for i in range(max_iter):
        x_curr = f(x_prev)
        
        # If the error is less than the tolerance, then it will converge.
        if np.abs(x_curr - x_prev) < tol:
            print(f"Converged in {i+1} iterations.")
            return x_curr
        x_prev = x_curr
    print("Max iterations reached. No convergence outside of the root.")
    return x_curr

# Example usage
g_func1 = lambda x: (3+x-2*x**2)**(1/4)
g_func2 = lambda x: ((x+3-x**4)/2)**(1/2)
g_func3 = lambda x: ((x+3)/(x**2+2))**(1/2)
g_func4 = lambda x: (3*x**4+2*x**2+3)/(4*x**3+4*x-1)
initial_guess = 1

fixed_point1 = fixed_point(g_func1, initial_guess)
fixed_point2 = fixed_point(g_func2, initial_guess)
fixed_point3 = fixed_point(g_func3, initial_guess)
fixed_point4 = fixed_point(g_func4, initial_guess)
print(f"The fixed point for g_1(x) is approximately: {fixed_point1:4f}")
print(f"The fixed point for g_2(x) is approximately: {fixed_point2:.4f}")
print(f"The fixed point for g_3(x) is approximately: {fixed_point3:.4f}")
print(f"The fixed point for g_4(x) is approximately: {fixed_point4:.4f}")
print("-" * 40)
# Exercise 2
def my_bisection_iterative(f, a, b, tol):
    if np.sign(f(a)) == np.sign(f(b)):
        raise Exception("The scalars a and b do not bound a root.")
        
    if f(a)*f(b) < 0:
        print(f"We can apply the IVT on {a} and {b}")
    elif f(a)*f(b) > 0:
        print(f"We cannot apply the IVT on {a} and {b}")
        

    # The condition is that the width of the interval should be greater than 2*tol
    # or loop while the error bound (b-a)/2 is greater than tol.
    while (b - a) / 2 > tol:
        m = (a + b) / 2
        # Check if m is the root, or determine which half to continue in
        if f(m) == 0:
            a = m
            b = m
            break
        elif np.sign(f(a)) == np.sign(f(m)):
            a = m
        else:
            b = m

    approx_root = (a + b) / 2
    # The error bound is half the width of the final interval
    error_bound = (b - a) / 2

    return approx_root, error_bound

f = lambda x: np.exp(x)-x**2+3*x-2
a = float(input("Enter the value for a: "))
b = float(input("Enter the value for b: "))

r001, err001 = my_bisection_iterative(f, a, b, 0.001)

print(f"r001 = {r001:.4f}, error_bound = {err001:.4f}")
