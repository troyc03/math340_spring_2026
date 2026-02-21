def my_nth_root(x, n, tol):
    """
    Calculates the n-th root of x using the Newton-Raphson method.

    Args:
        x (float): The number to find the root of (must be > 0).
        n (int): The degree of the root (must be > 1).
        tol (float): The tolerance for the error metric |f(y)| (must be > 0).

    Returns:
        float: An approximation of the n-th root of x.
    """
    if x <= 0 or tol <= 0 or n <= 1:
        raise ValueError("x and tol must be strictly positive scalars, and n must be an integer strictly greater than 1.")

    # Initial guess: a reasonable starting point is often 1 or x itself.
    # We can use x / n as a simple, often effective initial guess.
    r = x / n 
    
    # The derivative of f(y) = y^n - x is f'(y) = n * y^(n-1)
    # The Newton-Raphson formula is: r_new = r_old - f(r_old) / f'(r_old)
    # r_new = r - (r**n - x) / (n * r**(n-1))

    # Error metric is |f(y)| = |r**n - x|
    while abs(r**n - x) >= tol:
        f_r = r**n - x
        f_prime_r = n * r**(n - 1)
        r = r - f_r / f_prime_r
        
    return r

# --- Example Usage ---
x1 = 25.0
n1 = 2
tol1 = 1e-6
root1 = my_nth_root(x1, n1, tol1)
print(f"The {n1}-th root of {x1} is approximately: {root1}")
print(f"Verification: {root1**n1}") # Should be close to x1

print("-" * 20)

# Calculate the cube root of 27 with a tolerance of 1e-4
x2 = 27.0
n2 = 3
tol2 = 1e-4
root2 = my_nth_root(x2, n2, tol2)
print(f"The {n2}-th root of {x2} is approximately: {root2}")
print(f"Verification: {root2**n2}") # Should be close to x2

print("-" * 20)

# Calculate the 5th root of 100 with a tolerance of 1e-8
x3 = 100.0
n3 = 5
tol3 = 1e-8
root3 = my_nth_root(x3, n3, tol3)
print(f"The {n3}-th root of {x3} is approximately: {root3}")
print(f"Verification: {root3**n3}") # Should be close to x3

# Question 2
def fixed_point_iteration(g, x0, tol=1e-5, max_iter=100):
    """
    Performs fixed-point iteration to find a fixed point of function g.

    Args:
        g: The function for which to find the fixed point (g(x) = x).
        x0: The initial guess.
        tol: The convergence tolerance (stop when abs(x_new - x_old) < tol).
        max_iter: Maximum number of iterations to perform.

    Returns:
        The approximate fixed point.
    """
    x_old = x0
    for i in range(max_iter):
        x_new = g(x_old)
        if abs(x_new - x_old) < tol:
            print(f"Converged after {i+1} iterations.")
            return x_new
        x_old = x_new
    
    print("Did not converge within the maximum number of iterations.")
    return x_old

# Example Usage: Find the fixed point of g(x) = cos(x)
import numpy as np

# Define the function
g = lambda x: np.cos(x)

# Set initial guess
initial_guess = 0.1

# Find the fixed point
print("-" * 20)
fixed_point = fixed_point_iteration(g, initial_guess)
print(f"The fixed point is: {fixed_point}")


