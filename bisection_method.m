% Bisection Method
f = @(x) x^6-x-1; 
A = 1;
B = 2;
Delta = 10^-6;

% Compute initial condition values
YA = f(A); YB = f(B);
Max = 1 + floor(log(B-A)-log(Delta))/log(2);
if sign(YA) == sign(YB)
    fprintf("The values of f(a) and f(b) do not different in sign.\n")
    return; % Terminate algorithm
end

for K=1:Max
    C = (A+B)/2;
    YC = f(C);
    if YC == 0
        A=C; B=C;
    elseif sign(YB) == sign(YC)
            B=C; YB = YC;
    else
        A=C;
        YA = YC;
    end

    % Check for convergence
    if (B-A) < Delta
        true;
        break;
    end
end

fprintf('The computed root of f(x) = 0 is %.8f\n', C);
fprintf('The accuracy is +- %.8e\n', B - A);
fprintf('The value of the function f(C) is %.8e\n', YC);

