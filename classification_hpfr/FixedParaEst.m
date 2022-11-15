function [FixedPara, Sigb0]  = FixedParaEst(ydata, Tau, Amubeta, KU, sige, t, nn, B, D, niter)
% ============
% Description:
% ============
% This is the main function to estimate parameters of fixed term by
% Gibbs sampling.
%
% ============
% INPUT:
% ============
% ydata:           K*1 cell, the k-th cell includes the k-th traindata;
% Tau:             K*1 cell, each cell includes samples results using
%                  para_estimation_k.m
% Amubeta:         K*1 cell, each cell includes samples results of mu_beta
%                  using para_estimation_k.m
% KU:              K*1 cell, each cell includes samples results of kappa_u & kappa_v
%                  using para_estimation_k.m
% sige:            K*1 cell, each cell includes samples results of sige using
%                  para_estimation_k.m
% t:               m*1 vector, interval [tmin, tmax]
% nn:              The number of the ydata in different labels
% B:               basis functions; The default is B-spline.
% D:               nbasis of basis functions; The default is 14.
% niter:           number of samples; The default is 4000;

% ==========================================================================

if nargin < 10
    niter = 4000;
end

if nargin < 9
    D = 14;
end

if nargin < 8
    tmin = min(t);
    tmax = max(t);
    bbasis = create_bspline_basis([tmin, tmax], D , 4);
    B = eval_basis(t, bbasis);
end

%--------------
% Initialization
%--------------
a_1 = 0.1;
b_1 = 0.1;
sigb0 = 0.01;
m = length(t);

%--------------
% Sampling procedure
%--------------
K = size(Amubeta, 2);
FixedPara = cell(1, K);
Sigb0 = cell(1, K);
for k = 1 : K
    n = nn(k);
    KU1 = KU{k}(1, :);
    mubeta = Amubeta(:, k);
    tau = Tau{k};
    y = ydata{k};
    mubeta_mcmc = zeros(D, niter);
    sigb0_mcmc = zeros(m, niter);
    for iter = 1 : niter
        %--------------
        % Steps 4:
        % Sampling mu_beta0^(k) from the posterior
        % distribution p(mu_beta0^(k) | mu_beta^(k), sigma_b0^2(k)*I_p)
        %--------------
        mubeta0 = (mvnrnd(mubeta, sigb0 *eye(D), 1))';

        %         --------------
        % Steps 5:
        % Sampling mu_beta^(k) from the posterior
        % distribution p(mu_beta^(k) | tau, u, mu_beta0^(k), sigma_b0^2(k))
        %--------------
        ystretch = y(:);
        taustretch = tau(:);
        Bstretch = repmat(B, n, 1);
        invsigstretch = 1 / sige * kron(diag(1./KU1), eye(m));
        A2 = (1 / sigb0) * eye(D) + Bstretch' * invsigstretch * Bstretch;
        b = Bstretch' * invsigstretch * (ystretch - taustretch) +  (1 / sigb0) * eye(D) * mubeta0;
        [Utemp, Stemp] = svd(A2);
        invA2 = Utemp * diag(1./diag(Stemp)) * Utemp';
        invA2 = (invA2 + (invA2)') / 2;
        mubeta = mvnrnd(invA2 * b, invA2, 1);
        mubeta = mubeta(:);

        %--------------
        % Steps 6:
        % Sampling sigma_beta0^2(k) from the posterior
        % distribution p(sigma_beta0^2(k) | mu_beta^(k), mu_beta0^(k))
        %--------------
        sigb0 = gamrnd(D/2 + a_1, 2*b_1 / (2 + b_1 * (mubeta - mubeta0)' * (mubeta - mubeta0)), 1);
        sigb0 = 1/sigb0;


        mubeta_mcmc(:, iter) = mubeta;
        sigb0_mcmc(iter) = sigb0;
    end
    FixedPara{k} = mubeta_mcmc;
    Sigb0{k} = sigb0_mcmc;
end

end

