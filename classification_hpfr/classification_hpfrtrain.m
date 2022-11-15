function output_train = classification_hpfrtrain(input, distribution1, distribution2, nbreaks, norder, isample, niter)
% ============
% Description:
% ============
% This is the main function for parameter estimation using MCMC algorithom by Gibbs sampling based
% on the full conditional distributions of the parameters and latent
% variables. Steps are consistent with the paper, and details are provided
% in Section 3.3.
%
% ============
% INPUT:
% ============
% input: all the training data，including: traindata (each cell contains m*n_k matrix) , label;
% distribution1:   = 'N'or 'T'or 'S', the process of \epsilon(t));
% distribution2:   = 'N'or 'T'or 'S', the process of \tau(t); the default
%                  is the same as distribution1.
% norder:          order of B-splines (one higher than their degree);
% nbreaks:         number of breaks (also called knots), which are equally spaced in the time interval.
%                  The default is 14. Note that,
%                  nbasis = nbreaks + norder - 2 ;
% niter:           number of samples; The default is 4000;
% isample:         sampling points. The default is a sampling point of step 10 starting at half of the
%                  total number of samples.
%
% ==========================================================================

if nargin < 7
    niter = 4000;
end
if nargin < 6
    burnin = niter / 2; % The first burnin samples of burn-in period
    thin = 10;
    isample = (burnin+1) : thin : niter;
end
if nargin < 5
    nbreaks = 14;
end
if nargin < 4
    norder = 2;
end
if nargin < 3
    distribution2 = distribution1;
end


% Input data
K = length(input.traindata);
traindata = cell(K, 1);
nn = zeros(K, 1);
for k = 1 : K
    traindata{k} = input.traindata{k};
    nn(k) = size(traindata{k}, 2); % The number of the training data in different labels
end
m = size(traindata{1}, 1);
n = sum(nn);
tmin = 0;
tmax = 1;
t = linspace(tmin, tmax, m);
% Creat B spline basis functions
nbasis = nbreaks + norder - 2 ;
D = nbasis;
bbasis = create_bspline_basis([tmin, tmax], D , 4);
B = eval_basis(t, bbasis);

% --------Parameter Estimation--------

%--------------
% Step 1: Initialization
%--------------
% Initial value for hyperparameters in kernel functions
w = 1;
v0 = 0.04;
xi = log([w, v0]);
phi = normrnd(0, 1, 1, 2); % initial value for leapfrog algorithm

% Initial value for degrees
nu1 = 4;
nu2 = 4;

% Initial value for sigma_e^2
sige = 0.01;

% Initial value for sigma_beta^2(k)
sigb0 = 0.01 * ones(K, 1);

 
% ==========================================================================
% ------ Efficient method -------
% --------------
% Step 2:
% Sampling based on the full conditional distributions of
% the parameters and latent variables for the k-th class
% using the k-th data, respectively.
% --------------
meandata = cell(K, 1);
mubeta = cell(K, 1);
simul = cell(K, 1);
Tau = cell(K, 1);
Amubeta = [];
temp_sige = [];
Tau_k = cell(K, 1);
ydata = cell(K, 1);
KU = cell(K, 1);
for k = 1 : K
    meandata{k} = mean(traindata{k}, 2);
    mubeta{k} = regress(meandata{k}, B);
    [simul{k}, Tau{k}, KU{k}] = para_estimation_k(traindata{k}, mubeta{k}, t, distribution1, distribution2, sigb0(k), xi, phi, sige, nu1, nu2, B, D, niter);
    mubeta{k} = median(simul{k}(isample, 1 : D), 1)';
    Amubeta = cat(2, Amubeta, mubeta{k});
    temp_sige = cat(1, temp_sige, simul{k}(isample, D + 3));
    sige = median(temp_sige);
    Tau_k{k} = update_value_tau_k(Tau{k}, nn(k), isample);
    ydata{k} = traindata{k};
    KU{k} = update_value_KU(KU{k}, nn(k), isample);
end

% --------------
% Step 3:
% Update parameters of the fixed term with label k using k-th
% --------------
[FixedPara, Sigb0] = FixedParaEst(ydata, Tau_k, Amubeta, KU, sige, t, nn, B, D, niter);
ycenter = cell(K, 1);
Aycenter = [];
for k = 1 : K
    ycenter{k} = ydata{k} - kron(ones(1, nn(k)), B * mubeta{k});
    Aycenter = cat(2, Aycenter, ycenter{k});
end

% --------------
% Step 4:
% Update parameters of the random term
% --------------
[RandomPara, tau_mcmc, KU1_mcmc, KU2_mcmc] = RandomParaEst(Aycenter, distribution1, distribution2, t, n, m, D, niter);


% ==========================================================================
% %-------- Alternative method-------
% % The procedure for the alternative method is consistent with that in the
% % paper.
% % --------------
% % Initial value for priors on parameters: sigma_e^2, sigma_beta^2(k)
% a_0 = 0.1; b_0 = 0.1;
% a_1 = 0.1; b_1 = 0.1;
% KU1 = ones(n, 1);
% KU2 = ones(n, 1);
% for k = 1 : K
%     meandata{k} = mean(traindata{k}, 2);
%     mubeta{k} = regress(meandata{k}, B);
% end
% %--------------
% % Step 2 - 9: Sampling procedure
% %--------------
% FixedPara = cell(K, 1);
% Sigb0 = cell(K, 1);
% KU1_mcmc = cell(niter, 1);
% KU2_mcmc = cell(niter, 1);
% tau_mcmc = cell(niter, 1);
% RandomPara = zeros(niter, 5);
% for iter = 1 : niter
%     %--------------
%     % Step 2:
%     % Sampling tau_i from the posterior
%     % distribution p(tau_i | y_i, u_i, sigma_e^2, v_i, w, v0)
%     %--------------
%     ycenter = [];

%     for k = 1 : K
%         ycenter = cat(2, ycenter, ydata{k} - kron(ones(1, nn(k)), B * mubeta{k}));
%     end
%     tau = zeros(m,n);
%     covm = kernelfun(xi, t);
%     [U, S] = svd(covm);
%     for i = 1 : n
%         temp = 1/(sige * KU1(i)) + 1./ (diag(S) * KU2(i));
%         temp = 1./ temp; temp = diag(temp);
%         invA = U * temp * U';
%         invA = (invA + (invA)') / 2;
%         a = (1 / (KU1(i) * sige)) * ycenter(:, i);
%         tau(:, i) = (mvnrnd(invA * a, invA, 1))';
%     end
%
%
%     %--------------
%     % Step 3:
%     % Sampling sigma_e^2 from the posterior
%     % distribution p(sigma_e^2 | mu_beta, y_i, u_i)
%     %--------------
%     A = zeros(1, n);
%     for i =1 : n
%         A(i) =  (1/KU1(i)) * (yceter(:, i) - tau(:, i))' * (yceter(:, i) - tau(:, i));
%     end
%
%     sumA = sum(A);
%     sige = gamrnd(a_0 + m*n/2, ((2 * b_0) /((b_0 * sumA) + 2)) , 1);
%     sige = 1 / sige;
%
%     %--------------
%     % Step 4-6: Sampling fixed term
%     %--------------
%     temptau = tau;
%     tempKU1 = KU1;
%     mubeta0 = cell(K, 1);
%     for k = 1 : K
%         TEMPKU1 = tempKU1(1 : nn(k));
%         tempKU1(1 : nn(k)) = [];
%         %--------------
%         % Steps 4:
%         % Sampling mu_beta0^(k) from the posterior
%         % distribution p(mu_beta0^(k) | mu_beta^(k), sigma_b0^2(k)*I_p)
%         %--------------
%         mubeta0{k} = (mvnrnd(mubeta{k}, sigb0(k) * eye(D), 1))';
%
%         %--------------
%         % Steps 5:
%         % Sampling mu_beta^(k) from the posterior
%         % distribution p(mu_beta^(k) | tau, u, mu_beta0^(k), sigma_b0^2(k))
%         %--------------
%         ystretch = ydata{k}(:);
%
%         taustretch = temptau(:, 1 : nn(k));
%         taustretch = taustretch(:);
%         temptau(:, 1 : nn(k)) = [];
%         Bstretch = repmat(B, nn(k), 1);
%         invsigstretch = 1 / sige * kron(diag(1./TEMPKU1), eye(m));
%         A2 = (1 / sigb0(k)) * eye(D) + Bstretch' * invsigstretch * Bstretch;
%         b = Bstretch' * invsigstretch * (ystretch - taustretch) +  (1 / sigb0(k)) * eye(D) * mubeta0{k};
%         [Utemp, Stemp] = svd(A2);
%         invA2 = Utemp * diag(1./diag(Stemp)) * Utemp';
%         invA2 = (invA2 + (invA2)') / 2;
%         mubeta{k} = mvnrnd(invA2 * b, invA2, 1)';
%
%         %--------------
%         % Steps 6:
%         % Sampling sigma_beta0^2(k) from the posterior
%         % distribution p(sigma_beta0^2(k) | mu_beta^(k), mu_beta0^(k))
%         %--------------
%         sigb0temp = gamrnd(D/2 + a_1, 2*b_1 / (2 + b_1 * (mubeta{k} - mubeta0{k})' * (mubeta{k} - mubeta0{k})), 1);
%         sigb0(k) = 1/sigb0temp;
%     end
%
%     %--------------
%     % Steps 7:
%     % Update hyperparameters w, v0 in kernel functions
%     %--------------
%
%     [xi, phi] = Leapfrog(xi, phi, t, yceter, 1./KU1, 1./KU2, sige);
%
%     %--------------
%     % Steps 8:
%     % Sampling \KU1 & update \nu1
%     %--------------
%     U1 = ones(1, n);
%     U2 = ones(1, n);
%     KU1 = ones(1, n);
%     KU2 = ones(1, n);
%     if distribution1 == 'T'
%         bt1 = zeros(1, n);
%         for i = 1 : n
%             bt1(i) = nu1/2 + 1 / (2*sige) * (ycenter(:, i) - tau(:, i))' * (ycenter(:, i) - tau(:, i));
%             U1(i) = gamrnd((nu1 + m)/2, 1/bt1(i), 1); % sampling u_i from Gamma(a_1i, b_1i)
%             KU1(i) = 1 / U1(i); % obtain kappa(u_i)
%         end
%         %--------------
%         % Steps 8(a1):
%         % generate lambda1 from TG(2, nu_1)
%         %--------------
%         lambda1 = tgamrnd(0.02, 0.5, 2, 1/(nu1), 1, 1);
%         %--------------
%         % Steps 8(a2):
%         % generate nu_1 from conditional distribution p(nu_1 | u_i, lambda1)
%         % by MH path
%         %--------------
%         nu1 = MHnu( nu1, U1, lambda1, 'Hierar', 0.5); % generate nu_1 from
%     elseif distribution1 == 'S'
%         bt1 = zeros(1, n);
%         for i = 1 : n
%             bt1(i) = 1 / (2*sige) * (ycenter(:, i) - tau(:, i))' * (ycenter(:, i) - tau(:, i));
%             U1(i) = tgamrnd(0, 1, nu1 + m/2, 1/bt1(i), 1); % sampling u_i from Beta(a_1i, b_1i)
%             KU1(i) = 1 / U1(i); % obtain kappa(u_i)
%         end
%         %--------------
%         % Steps 8(b1):
%         % generate lambda1 from TG(2, nu_1)
%         %--------------
%         lambda1 = tgamrnd(0.02, 0.5, 2, 1/(nu1), 1, 1);
%         %--------------
%         % Steps 8(b2):
%         % generate nu_1 from conditional distribution p(nu_1 | u_i, lambda1)
%         % by MH path
%         %--------------
%         nu1 = MHnu(nu1, U1, lambda1, 'Hierar', 0.5); % generate nu_1 from
%     elseif distribution1 == 'N'
%         KU1 = ones(1, n);
%     end
%
%     %--------------
%     % Steps 9:
%     % Sampling \KU2 & update \nu2
%     %--------------
%     if distribution2 == 'T'
%         s = 1e-8;
%         [U, S] = svd(covm + s * eye(m));
%         invcovm =  U * diag(1./ diag(S)) * U';
%         invcovm = (invcovm + invcovm')/2;
%         bt2 = zeros(1, n);
%         for i = 1 : n
%             bt2(i) = nu2/2 + 0.5*(tau(:, i))' * invcovm *tau(:, i);
%             U2(i) = gamrnd((nu2 + m)/2, 1/bt2(i), 1); % sampling v_i from Gamma(a_2i, b_2i)
%             KU2(i) = 1 / U2(i); % obtain kappa(v_i)
%         end
%         %--------------
%         % Steps 9(a1):
%         % generate lambda2 from TG(2, nu_2)
%         %--------------
%         lambda2 = tgamrnd(0.02, 0.5, 2, 1/(nu2), 1, 1); % generate lambda2 from TG(2, nu_2)
%         %--------------
%         % Steps 9(a2):
%         % generate nu_2 from conditional distribution p(nu_2 | v_i, lambda2)
%         % by MH path
%         %--------------
%         nu2 = MHnu(nu2, U2, lambda2, 'Hierar',0.5);
%     elseif distribution2 == 'S'
%         s = 1e-8;
%         [U, S] = svd(covm + s * eye(m));
%         invcovm =  U * diag(1./ diag(S)) * U';
%         invcovm = (invcovm + invcovm')/2;
%         bt2 = zeros(1, n);
%         for i = 1 : n
%             bt2(i) = 0.5 * (tau(:, i))' * invcovm* tau(:, i);
%             U2(i) = tgamrnd(0, 1, nu2 + D/2, 1/bt2(i), 1);% sampling v_i from Beta(a_2i, b_2i)
%             KU2(i) = 1 / U2(i); % obtain kappa(v_i)
%         end
%         %--------------
%         % Steps 9(b1):
%         % generate lambda2 from TG(2, nu_2)
%         %--------------
%         lambda2 = tgamrnd(0.02, 0.5, 2, 1/(nu2), 1, 1); % generate lambda2 from TG(2, nu_2)
%         %--------------
%         % Steps 9(b2):
%         % generate nu_2 from conditional distribution p(nu_2 | v_i, lambda2)
%         % by MH path
%         %--------------
%         nu2 = MHnu(nu2, U2, lambda2, 'Hierar',0.5);
%     elseif distribution2 == 'N'
%         KU2 = ones(1, n);
%     end
%
%     for k = 1 : K
%         FixedPara{k}(:, iter) = mubeta{k};
%         Sigb0{k}(:, iter) = sigb0(k);
%     end
%     KU1_mcmc{iter} = KU1;
%     KU2_mcmc{iter} = KU2;
%     tau_mcmc{iter} = tau;
%     RandomPara(iter,:) = [exp(xi), sige, nu1, nu2];
% end
% ==========================================================================



output_train.FixedPara = FixedPara;
output_train.Sigb0 = Sigb0;
output_train.KU1 = KU1_mcmc;
output_train.KU2 = KU2_mcmc;
output_train.RandomPara = RandomPara;
output_train.ATau = tau_mcmc;
output_train.m = m;
output_train.B = B;




