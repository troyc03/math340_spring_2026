import numpy as np

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

f = lambda x: x**6 - x - 1
a = float(input("Enter the value for a: "))
b = float(input("Enter the value for b: "))

r001, err001 = my_bisection_iterative(f, a, b, 0.001)

print(f"r001 = {r001:.4f}, error_bound = {err001:.4f}")
print(f"f(r001) = {f(r001):.4e}")
