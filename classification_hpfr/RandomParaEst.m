function [RandomPara, Tau, KU_1, KU_2] = RandomParaEst(y, distribution1, distribution2, t, n, m, D, niter)
% ============
% Description:
% ============
% This is the main function to estimate parameters of random term by
% Gibbs sampling. Steps are consistent with the paper, and 
% details are provided in Section 3.3
% 
% ============
% INPUT:
% ============
% y:               m*n matrix, y centralized;
% distribution1:   = 'N'or 'T'or 'S', the process of \epsilon(t));
% distribution2:   = 'N'or 'T'or 'S', the process of \tau(t); the default
%                  is the same as distribution1.
% m:               The number of observations of subject y_i;
% n:               The number of subjects y;
% t:               m*1 vector, interval [tmin, tmax]
% D:               nbasis of basis functions; The default is 14;
% niter:           number of samples; The default is 4000.

% ==========================================================================
if nargin < 8
    niter = 4000;
end

if nargin < 7
    D = 14;
end

if nargin < 6
    [m, ~] = size(y);
end

if nargin < 5
    [m, n] = size(y);
end

if nargin < 4
    tmin = 0;
    tmax = 1;
    t = linspace(tmin, tmax, m);
end

if nargin < 4
    distribution2 = distribution1;
end

% Initial value for hyperparameters in kernel functions
w = 1;
v0 = 0.04;
xi = log([w, v0]);
phi = normrnd(0, 1, 1, 2); % Initial value for leapfrog algorithm

% Initial value for sigma_e^2
sige = 0.01;

% Initial value for u, v, kappa_u, kappa_v, nu_1 and nu_2
U1 = ones(1, n); U2 = ones(1, n);
KU1 = ones(1, n); KU2 = ones(1, n);
nu1 = 4; nu2 = 4;
% Initial value for a_0 and b_0
a_0 = 0.1; b_0 = 0.1;


for iter = 1 : niter
    %--------------
    % Step 2:
    % Sampling tau_i from the posterior
    % distribution p(tau_i | y_i, u_i, sigma_e^2, v_i, w, v0)
    %--------------
    tau = zeros(m,n);
    covm = kernelfun(xi, t);
    [U, S] = svd(covm);
    for i = 1 : n
        temp = 1/(sige * KU1(i)) + 1./ (diag(S) * KU2(i));
        temp = 1./ temp; temp = diag(temp);
        invA = U * temp * U';
        invA = (invA + (invA)') / 2;

        a = (1 / (KU1(i) * sige)) * y(:, i);
        tau(:, i) = (mvnrnd(invA * a, invA, 1))';
    end
    %--------------
    % Step 3:
    % Sampling sigma_e^2 from the posterior
    % distribution p(sigma_e^2 | mu_beta, y_i, u_i)
    %--------------
    A1 = zeros(1, n);
    for i =1 : n
        A1(i) =  (1/KU1(i)) * (y(:, i) - tau(:, i))' * (y(:, i) - tau(:, i));
    end
    sumA1 = sum(A1);
    sige = gamrnd(a_0 + m*n/2, ((2 * b_0) /((b_0 * sumA1) + 2)) , 1);
    sige = 1 / sige;
    %--------------
    % Steps 7:
    % Update hyperparameters w, v0 in kernel functions
    %--------------

    [xi, phi] = Leapfrog(xi, phi, t, y, U1, U2, sige);
    %--------------
    % Steps 8:
    % Sampling \KU1 & update \nu1
    %--------------
    if distribution1 == 'T'
        bt1 = zeros(1, n);
        for i = 1 : n
            bt1(i) = nu1/2 + 1 / (2*sige) * (y(:, i) - tau(:, i))' * (y(:, i) - tau(:, i));
            U1(i) = gamrnd((nu1 + m)/2, 1/bt1(i), 1);%V1
            KU1(i) = 1 / U1(i);
        end
        %%update \nu1
        lambda1 = tgamrnd(0.02, 0.5, 2, 1/(nu1), 1, 1);
        nu1 = MHnu( nu1, U1, lambda1, 'Hierar', 0.5);

    elseif distribution1 == 'S'
        bt1 = zeros(1, n);
        for i = 1 : n
            bt1(i) = 1 / (2*sige) * (y(:, i) - tau(:, i))' * (y(:, i) - tau(:, i));
            U1(i) = tgamrnd(0, 1, nu1 + m/2, 1/bt1(i), 1);%V1
            KU1(i) = 1 / U1(i);
        end
    elseif distribution1 == 'N'

        KU1 = ones(1, n);
    end

    %--------------
    % Steps 9:
    % Sampling \KU2 & update \nu2
    %--------------
    if distribution2 == 'T'
        s = 1e-8;
        [U, S] = svd(covm + s * eye(m));
        invcovm =  U * diag(1./ diag(S)) * U';
        invcovm = (invcovm + invcovm')/2;
        bt2 = zeros(1, n);
        for i = 1 : n
            bt2(i) = nu2/2 + 0.5*(tau(:, i))' * invcovm *tau(:, i);
            U2(i) = gamrnd((nu2 + m)/2, 1/bt2(i), 1);%V2
            KU2(i) = 1 / U2(i);
        end
        %--------------
        % Steps 9(a1):
        % generate lambda2 from TG(2, nu_2)
        %--------------
        lambda2 = tgamrnd(0.02, 0.5, 2, 1/(nu2), 1, 1);
        %--------------
        % Steps 9(a2):
        % generate nu_2 from conditional distribution p(nu_2 | v_i, lambda2)
        % by MH path
        %--------------
        nu2 = MHnu( nu2, U2, lambda2, 'Hierar',0.5);

    elseif distribution2 == 'S'
        s = 1e-8;
        [U, S] = svd(covm + s * eye(m));
        invcovm =  U * diag(1./ diag(S)) * U';
        invcovm = (invcovm + invcovm')/2;
        bt2 = zeros(1, n);
        for i = 1 : n
            bt2(i) = 0.5 * (tau(:, i))' * invcovm* tau(:, i);
            U2(i) = tgamrnd(0, 1, nu2 + p/2, 1/bt2(i), 1);
            KU2(i) = 1 / U2(i);
        end
        %--------------
        % Steps 9(b1):
        % generate lambda2 from TG(2, nu_2)
        %--------------
        lambda2 = tgamrnd(0.02, 0.5, 2, 1/(nu2), 1, 1); % generate lambda2 from TG(2, nu_2)
        %--------------
        % Steps 9(b2):
        % generate nu_2 from conditional distribution p(nu_2 | v_i, lambda2)
        % by MH path
        %--------------
        nu2 = MHnu(nu2, U2, lambda2, 'Hierar',0.5);
    elseif distribution2 == 'N'
        KU2 = ones(1, n);
    end


    KU_1{iter} = KU1;
    KU_2{iter} = KU2;
    Tau{iter} = tau;
    RandomPara(iter,:) = [exp(xi), sige, nu1, nu2];
end

end

