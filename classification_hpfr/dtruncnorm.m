function [y] = dtruncnorm(num,lower,upper ,mean,sd)
%upper为截断上边界，lower为截断下边界\
% F = @(x)(normpdf(x,mean,sd));
% a = integral(F, lower, upper);

a = normcdf(upper,mean,sd) - normcdf(lower,mean,sd);
y = (1/a) * normpdf(num,mean,sd);

% pd = makedist('Normal','mu',mean,'sigma',sd);
% t = truncate(pd,lower,upper);
% y = pdf(t,num);
end

