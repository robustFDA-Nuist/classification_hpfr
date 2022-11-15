function covm  = kernelfun(xi, t)
% This is the main function for kernel functions.

expxi = exp(xi);
m = length(t);
t = t(:); % Convert time t to a column vector

covm = exp(-0.5 * expxi(1) * (t * ones(1, m) - ones(m, 1) * (t)') .^2);
covm = expxi(2) * covm;
end

