import numpy as np
from scipy.optimize import newton
import matplotlib.pyplot as plt

# =============================
# --- Exercise 1 Solution ---
# =============================
def solve_nose_in_angle(l, h, D, beta1_deg, initial_guess_deg=30.0):
    beta1 = np.radians(beta1_deg)
    
    # Constants based on vehicle dimensions
    A = l * np.sin(beta1)
    B = l * np.cos(beta1)
    C = (h + 0.5 * D) * np.sin(beta1) + 0.5 * D * np.tan(beta1)
    E = (h + 0.5 * D) * np.cos(beta1) - 0.5 * D

    # Solver works in radians
    def equation(alpha):
        return (A * np.sin(alpha) * np.cos(alpha)) + B * (np.sin(alpha)**2) - C * np.cos(alpha) - E * np.sin(alpha)
    
    # Newton's method (guess converted to radians)
    beta_solution_rad = newton(equation, np.radians(initial_guess_deg))
    return np.degrees(beta_solution_rad)

L, H, D, beta1 = 89, 49, 30, 11.5
alpha_1 = solve_nose_in_angle(L, H, D, beta1)
print(f"Exercise 1: For D = {D} in, alpha is: {alpha_1:.2f} degrees")

# Plot results
alpha_range = np.radians(np.linspace(0, 90, 100))
beta1_rad = np.radians(beta1)
A = L * np.sin(beta1_rad)
B = L * np.cos(beta1_rad)
C = (H + 0.5 * D) * np.sin(beta1_rad) + 0.5 * D * np.tan(beta1_rad)
E = (H + 0.5 * D) * np.cos(beta1_rad) - 0.5 * D
equation_values = A * np.sin(alpha_range) * np.cos(alpha_range) + B * (np.sin(alpha_range)**2) - C * np.cos(alpha_range) - E * np.sin(alpha_range)
plt.plot(np.degrees(alpha_range), equation_values, label='Equation Value')
plt.axhline(0, color='red', linestyle='--', label='Zero Line')
plt.scatter(alpha_1, 0, color='green', label=f'Solution: {alpha_1:.2f}°')
plt.title('Nose-in Angle Equation')
plt.xlabel('Alpha (degrees)')
plt.ylabel('Equation Value')
plt.legend()
plt.grid(True)
plt.show()

# =============================
# --- Exercise 2 Solution ---
# =============================
g = 32.17
m = 0.25
k = 0.1
x0 = 300 

def x(t):
    # Standard terminal velocity position equation
    return x0 - (m*g/k)*t + (m**2 * g/k**2)*(1 - np.exp(-k*t/m))

def v(t):
    # Derivative of x(t)
    return -(m*g/k) + (m*g/k)*np.exp(-k*t/m)

# Use a smaller initial guess (e.g., 5 seconds) for better convergence
t_impact = newton(x, 5.0, fprime=v)
print(f"Exercise 2: Impact time is: {t_impact:.2f} seconds")

# --- Visualization ---
t_plot = np.linspace(0, t_impact, 100)
plt.plot(t_plot, x(t_plot), color='green')
plt.axhline(0, color='red', linestyle='--')
plt.scatter(t_impact, 0, color='blue', label=f'Impact Time: {t_impact:.2f} s')
plt.title('Position vs Time for Falling Object')
plt.xlabel('Time (s)')
plt.ylabel('Position (ft)')
plt.grid(True)
plt.show()

# =============================
# --- Exercise 3 Solution ---
# =============================

# Constants
omega = 7.2921e-5
g = 32.17
x = 1.5
C = (2 * omega**2 * x)/g

def f(u):
    return np.sinh(u) - np.sin(u) - C
def df(u):
    return np.cosh(u) - np.cos(u)

# Newton's method implementation
def newtons_method(f, df, initial_guess, tol=1e-6, max_iter=100):
    u = initial_guess
    for _ in range(max_iter):
        f_u = f(u)
        df_u = df(u)
        if df_u == 0:  # Avoid division by zero
            raise ValueError("Derivative is zero. No solution found.")
        u_new = u - f_u / df_u
        if abs(u_new - u) < tol:
            return u_new
        u = u_new
    raise ValueError("Maximum iterations reached. No solution found.")

# Secant Method implementation
def secant_method(f, x0, x1, tol=1e-6, max_iter=100):
    for _ in range(max_iter):
        f_x0 = f(x0)
        f_x1 = f(x1)
        if f_x1 - f_x0 == 0:  # Avoid division by zero
            raise ValueError("Function values are the same. No solution found.")
        x2 = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
        if abs(x2 - x1) < tol:
            return x2
        x0, x1 = x1, x2
    raise ValueError("Maximum iterations reached. No solution found.")

# Fixed point iteration implementation
def fixed_point_iteration(g, initial_guess, tol=1e-6, max_iter=100):
    u = initial_guess
    for _ in range(max_iter):
        u_new = g(u)
        if abs(u_new - u) < tol:
            return u_new
        u = u_new
    raise ValueError("Maximum iterations reached. No solution found.")

# Example for Newton's
initial_guess = 1.7
u_solution = newtons_method(f, df, initial_guess)
print(f"Exercise 3: Solution using Newton's method is: {u_solution:.6f}")

# Example for Secant Method
x0, x1 = 0.5, 1.5
u_solution_secant = secant_method(f, x0, x1)
print(f"Exercise 3: Solution using Secant method is: {u_solution_secant:.6f}")

# Example for Fixed Point Iteration
def g(u):
    return np.arcsinh(np.sin(u) + C)
initial_guess_fp = 0.001
u_solution_fp = fixed_point_iteration(g, initial_guess_fp)
print(f"Exercise 3: Solution using Fixed Point Iteration is: {u_solution_fp:.6f}")

# Plot results for Exercise 3
u_range = np.linspace(0, 2, 100)
plt.plot(u_range, f(u_range), label='f(u)')
plt.axhline(0, color='red', linestyle='--', label='Zero Line')
plt.scatter(u_solution, 0, color='green', label=f'Newton Solution: {u_solution:.6f}')
plt.scatter(u_solution_secant, 0, color='blue', label=f'Secant Solution: {u_solution_secant:.6f}')
plt.scatter(u_solution_fp, 0, color='orange', label=f'Fixed Point Solution: {u_solution_fp:.6f}')
plt.title('Function f(u) for Exercise 3')
plt.xlabel('u')
plt.ylabel('f(u)')
plt.legend()
plt.grid(True)
plt.show()

