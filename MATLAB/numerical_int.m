% Midpoint Rule

a = 2; b = 4; n = 4;
dx = (b - a) / n;
f = @(x) 1 ./ log(x);
midpoints = (a + dx/2) : dx : (b - dx/2);
approximation = dx * sum(f(midpoints));
fprintf('The approximation is: %.4f\n', approximation);

% Trapezoidal Rule
f = @(x) cos(x)./x; a=1; b=4; n = 8;
dx = (b-a)/n; x = a:h:b;
y = f(x);
% Sum all y values, but subtract half of the first and last points
integral_val = h * (sum(y) - (y(1) + y(end))/2);
result = trapz(x, y);
fprintf('The approximation is: %.2f\n', integral_val);
fprintf('The approximation using trapz() is: %.2f\n', result);

% Simpson's Rule
f = @(x) cos(x)./x;
a = 1;
b = 6; 
n = 4; % Number of points
h = (b-a)/n;
x = a:h:b;      % grid points
y = f(x);      % function values
I = (h/3) * ( y(1) + y(end) ...
    + 4*sum(y(2:2:end-1)) ...
    + 2*sum(y(3:2:end-2)) );
fprintf('The approximation is: %.6f\n', I);