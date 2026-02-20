# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 15:34:49 2026

@author: Troy
"""

import numpy as np

def newtons_method_recursive(f, df, x0, tol):
    if abs(f(x0)) < tol:
        return x0
    else:
        return newtons_method_recursive(f, df, x0 - f(x0)/df(x0), tol)

f = lambda x: x**2-2
f_prime = lambda x: 2*x
estimate = newtons_method_recursive(f, f_prime, 1.5, 1e-6)
print()
print("RECURSIVE METHOD")
print("estimate =", estimate)
print()

def newtons_method_iterative(f, df, x0, tol, max_iterations=1000):
    x = x0
    for _ in range(max_iterations):
        fx = f(x)
        if abs(fx) < tol:
            return x
        dfx = df(x)
        if dfx == 0:
            raise ValueError("Derivative is zero. Cannot continue Newton's method.")
        x = x - fx / dfx
    
    # If the method doesn't converge within max_iterations, return the last estimate
    print(f"Warning: Newton's method did not converge within {max_iterations} iterations.")
    return x

f = lambda x: np.exp(x)-x
f_prime = lambda x: np.exp(x)-1

estimate = newtons_method_iterative(f, f_prime, 1.5, 1e-6)
print()
print("ITERATIVE METHOD")
print("estimate =", estimate)
print()
