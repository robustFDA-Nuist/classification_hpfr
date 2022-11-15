function postpredpro = postpredpro(y, sige, theta, B, mubeta, isample, c, t, obpoint)
% ============
% Description:
% ============
% This is the main function for computing posterior predictive probability.
%
% ============
% INPUT:
% ============
% y:       observations of the test subject
% sige:    parameter sigma_e^2;
% theta:   hyperparameters w, v0 in kernel functions
% B:       n*p matrix, basis functions
% mubeta:  K*1 cell, each cell has a vector of basis coefficients for the k-th
%          class
% isample: sampling points;  
% c:       class
% t:       interval of time, t 
% obpoint: positions of observations of the test subject in interval of time, t


% ==========================================================================

M = length(isample);
m = length(obpoint);
y_postpredpro1 = zeros(M, 1);
j = 1;
for i = isample
    covm = kernelfun(log(theta(i, :)), t);
    y_postpredpro1(i) = mvnpdf(y, B(obpoint, :) * mubeta{c}(:, i), sige(i) * eye(m) + covm(obpoint, obpoint));
    j = j + 1;
end
postpredpro = mean(y_postpredpro1);
end

