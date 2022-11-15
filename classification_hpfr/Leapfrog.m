function [xi, phi] = Leapfrog(xi, phi, t, y, U1, U2, sige)
% This is the main function for 'Leapfrog' algorithm.

m = size(y, 1);
n = size(y, 2);
Nm = size(y, 1);
epsilon = 0.5 * Nm^(-0.5);
lambda = 1;
alpha = 0.95;
[fx, dfx] = exppdc(xi, t, y, U1, U2, sige);
    halfphi = phi - epsilon/2 * dfx;
    newxi =xi +  epsilon/lambda * halfphi;
[newfx, newdfx] = exppdc(newxi, t, y, U1, U2, sige);
    newphi = halfphi - epsilon/2 * newdfx;
r = exp(fx - newfx + (0.5/lambda) *sum(newphi) - (0.5/lambda) *sum(phi));
if (unifrnd(0, 1) < min(1, r))
    xi = newxi;
    phi = newphi;
else
    phi = -1 * phi;
end
    v = normrnd(0, 1, 1, length(phi));
    phi = alpha * phi + sqrt(1 - alpha^2) * v;
end

