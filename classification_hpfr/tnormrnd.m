function [x] = tnormrnd(a,b,alp,bet,m,n)
% generate m times n random numbers from truncated normal distribution
% gamma(alp,bet) such that a<x<b

if nargin == 4
    m = 1; n = 1;
end
if nargin == 5
    n = m;
end

% pd = makedist('Normal','mu',alp,'sigma',bet);
% t = truncate(pd, a, b);
% x = random(t, m, n);

u=rand(m,n);
interval = normcdf([a b],alp,bet);
v=interval(1)+(interval(2)-interval(1))*u;
x=norminv(v,alp,bet);
end
