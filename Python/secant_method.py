import numpy as np

def secant_method(f, x0, x1, tol=1e-10, max_iter=100):
    """
    Finds a root of f(x) = 0 using the Secant Method.
    """
    # Evaluate initial function values
    f_x0 = f(x0)
    f_x1 = f(x1)

    for i in range(max_iter):
        # Check if denominator is too small to avoid division by zero
        if abs(f_x1 - f_x0) < 1e-15: # Using a smaller, strict epsilon for safety
            # If the slope is nearly flat, we can't continue
            raise ValueError("Secant method failed: denominator close to zero.")

        # Compute the next approximation
        # Formula: x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        x2 = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)

        # Check for convergence
        if abs(f(x2)) < tol:
            return x2, i + 1
        
        # Update values for next iteration
        x0, x1 = x1, x2
        f_x0 = f_x1
        f_x1 = f(x2)

    raise ValueError("Secant method did not converge within the maximum number of iterations.")

# Define the function and initial guesses
f = lambda x: np.exp(x) - x**2
x0 = 0.0
x1 = 1.0

# Run the method
try:
    estimate, iterations = secant_method(f, x0, x1, tol=1e-10)
    print(f"Secant method estimate: {estimate:.6f} found in {iterations} iterations")
except ValueError as e:
    print(e)
