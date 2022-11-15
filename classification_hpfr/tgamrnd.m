function [x] = tgamrnd(a,b,alp,bet,m,n)
% generate m times n random numbers from truncated gamma distribution
% gamma(alp,bet) such that a<x<b
% see in Lachos, et al. (Statistica Sinica, 2010, Appendix B)
% a,b为截断范围
% alp bet 为参数:shape and scale


if nargin == 4
    m = 1; n = 1;
end
if nargin == 5
    n = m;
end

% pd = makedist('gamma','a',alp,'b',bet);
% t = truncate(pd, a, b);
% x = random(t, m, n);
u=rand(m,n);
interval = gamcdf([a b],alp,bet);
v=interval(1)+(interval(2)-interval(1))*u;
x=gaminv(v,alp,bet);

end