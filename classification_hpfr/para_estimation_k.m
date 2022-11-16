function [simul, Tau, KU] = para_estimation_k(y, mubeta, t, distribution1, distribution2, sigb0, xi, phi, sige, nu1, nu2, B, D, niter)
% ============
% Description:
% ============
% This is the main function to sample based on the full conditional distributions of
% the parameters and latent variables for the k-th class
% using the k-th data, respectively.
%
% ============
% INPUT:
% ============
% y:               m*n matrix, y centralized;
% mubeta:          initial values of mu_beta
% t:               m*1 vector, interval [tmin, tmax]
% distribution1:   = 'N'or 'T'or 'S', the process of \epsilon(t));
% distribution2:   = 'N'or 'T'or 'S', the process of \tau(t); the default
%                  is the same as distribution1.
% sigb0,xi,phi,sige,nu1,nu2: initial values of sigma_b0^2,xi,phi,sigma_e^2,nu1,nu2;
% B:               basis functions; The default is B-spline.
% D:               nbasis of basis functions; The default is 14;
% niter:           number of samples; The default is 4000.

% ==========================================================================

if nargin < 14
    niter = 4000;
end

if nargin < 13
    D = 14;
end

if nargin < 12
    tmin = min(t);
    tmax = max(t);
    bbasis = create_bspline_basis([tmin, tmax], D , 4);
    B = eval_basis(t, bbasis);
end

if nargin < 11
    nu2 = 4;
end

if nargin < 10
    nu1 = 4;
end

if nargin < 9
    sige = 0.01;
end

if nargin < 8
    phi = normrnd(0, 1, 1, 2);
end

if nargin < 7
    w = 1;
    v0 = 0.04;
    xi = log([w, v0]);
end

if nargin < 6
    sigb0 = 0.01;
end

if nargin < 5
    distribution2 = distribution1;
end


% --------------
% Step1 :
% Initial values
% --------------
a_0 = 0.1;
b_0 = 0.1;
a_1 = 0.1;
b_1 = 0.1;
n = size(y, 2);
simul = zeros(niter, D+6);
m = length(t);
U1 = ones(1, n);
U2 = ones(1, n);
KU1 = ones(1, n);
KU2 = ones(1, n);
Tau = cell(niter, 1);

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
        %inv(A)
        %--------- mehtod 1-----------------------------
        %          temp = diag(1 ./ (1 + KU2(i) / (sige * KU1(i)) .* diag(S)));
        %          temp = U * temp * U';
        %          invA = KU1(i) * sige * (eye(m) - temp);
        %          invA = (invA + (invA)') / 2;
        %--------------------------------------
        %--------- mehtod 2-----------------------------
        temp = 1/(sige * KU1(i)) + 1./ (diag(S) * KU2(i));
        temp = 1./ temp; temp = diag(temp);
        invA = U * temp *U';
        invA = (invA + (invA)') / 2;
        %--------------------------------------

        a = (1 / (KU1(i) * sige)) * (y(:, i) - B * mubeta);
        tau(:, i) = (mvnrnd(invA * a, invA, 1))';
    end
    %--------------
    % Step 3:
    % Sampling sigma_e^2 from the posterior
    % distribution p(sigma_e^2 | mu_beta, y_i, u_i)
    %--------------
    A1 = zeros(1, n);
    for i =1 : n
        A1(i) =  (1/KU1(i)) * (y(:, i) - B * mubeta - tau(:, i))' * (y(:, i) - B * mubeta - tau(:, i));
    end
    sumA1 = sum(A1);
    sige = gamrnd(a_0 + m*n/2, ((2 * b_0) /((b_0 * sumA1) + 2)) , 1);
    sige = 1 / sige;

    %--------------
    % Step 4-6: Sampling fixed term
    %--------------
    %--------------
    % Steps 4:
    % Sampling mu_beta0^(k) from the posterior
    % distribution p(mu_beta0^(k) | mu_beta^(k), sigma_b0^2(k)*I_p)
    %--------------
    mubeta0 = (mvnrnd(mubeta, sigb0 *eye(D), 1))';
    %--------------
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
    %--------------
    % Steps 7:
    % Update hyperparameters w, v0 in kernel functions
    %--------------

    ycenter = y - kron(ones(1, n), B * mubeta);
    [xi, phi] = Leapfrog(xi, phi, t, ycenter, U1, U2, sige);
    %--------------
    % Steps 8:
    % Sampling \KU1 & update \nu1
    %--------------
    if distribution1 == 'T'
        bt1 = zeros(1, n);
        for i = 1 : n
            bt1(i) = nu1/2 + 1 / (2*sige) * (y(:, i) - B * mubeta - tau(:, i))' * (y(:, i) - B*mubeta - tau(:, i));
            U1(i) = gamrnd((nu1 + m)/2, 1/bt1(i), 1);%V1
            KU1(i) = 1 / U1(i);
        end
        %--------------
        % Steps 8(a1):
        % generate lambda1 from TG(2, nu_1)
        %--------------
        lambda1 = tgamrnd(0.02, 0.5, 2, 1/(nu1), 1, 1);
        %--------------
        % Steps 8(a2):
        % generate nu_1 from conditional distribution p(nu_1 | u_i, lambda1)
        % by MH path
        %--------------
        nu1 = MHnu( nu1, U1, lambda1, 'Hierar',0.5);
    elseif distribution1 == 'S'
        nu1 = 4;
        nu2 = 4;
        bt1 = zeros(1, n);
        for i = 1 : n
            bt1(i) = 1 / (2*sige) * (y(:, i) - B * mubeta - tau(:, i))' * (y(:, i) - B*mubeta - tau(:, i));
            U1(i) = tgamrnd(0, 1, nu1 + m/2, 1/bt1(i), 1);%V1
            KU1(i) = 1 / U1(i);
        end
        %--------------
        % Steps 8(b1):
        % generate lambda2 from TG(2, nu_1)
        %--------------
        lambda1 = tgamrnd(0.02, 0.5, 2, 1/(nu1), 1, 1); % generate lambda2 from TG(2, nu_2)
        %--------------
        % Steps 8(b2):
        % generate nu_1 from conditional distribution p(nu_1 | U_i, lambda1)
        % by MH path
        %--------------
        nu1 = MHnu(nu1, U1, lambda1, 'Hierar',0.5);
    elseif distribution1 == 'N'
        %
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
            bt2(i) = nu2/2 + (tau(:, i))' * invcovm *tau(:, i);
            U2(i) = gamrnd((nu2 + D)/2, 1/bt2(i), 1);%V2
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
            U2(i) = tgamrnd(0, 1, nu2 + D/2, 1/bt2(i), 1);
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

    %% output
    Tau{iter} = tau;
    simul(iter,:) = [(mubeta)', exp(xi), sige, nu1, nu2, sigb0];
    KU_1{iter} = KU1;
    KU_2{iter} = KU2;
    KU{1} = KU_1;
    KU{2} = KU_2;
end
end

