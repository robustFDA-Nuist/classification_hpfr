function [y] = dtruncnorm(num,lower,upper ,mean,sd)
%upperΪ�ض��ϱ߽磬lowerΪ�ض��±߽�\
% F = @(x)(normpdf(x,mean,sd));
% a = integral(F, lower, upper);

a = normcdf(upper,mean,sd) - normcdf(lower,mean,sd);
y = (1/a) * normpdf(num,mean,sd);

% pd = makedist('Normal','mu',mean,'sigma',sd);
% t = truncate(pd,lower,upper);
% y = pdf(t,num);
end

