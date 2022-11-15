function output_train = training(input, K, p, Distribution1, Distribution2, niter, isample)
% ============
% Description:
% ============
% This is the main function for parameter estimation.
% Includes the following steps:
%



%% Initialization parameters
w = 1;
v0 = 0.04;
xi = log([w, v0]); % Hyperparameters in kernel functions
nu1 = 4;
nu2 = 4;
phi = normrnd(0, 1, 1, 2);
sige = 0.01;
sigb0 = 0.01;

% Input data
S_work = cell(K, 1);
nn = zeros(K, 1);
n_c = [];
for k = 1 : K
    S_work{k} = input.S_work{k};
    nn(k) = size(S_work{k}, 2);
    n_c = [n_c nn(k)];
end
% The number of the training data in different labels
m = size(S_work{1}, 1); % The number of observations per subject
n = sum(nn);

% Creat B spline basis functions
tmin = 0;
tmax = 1;
t = linspace(tmin, tmax, m);
bbasis = create_bspline_basis([tmin, tmax], p , 4);
B = eval_basis(t, bbasis);


%% Initialization
meanS_work = cell(K, 1);
mubetaS_work = cell(K, 1);
simulS_work = cell(K, 1);
TauS_work = cell(K, 1);
KUS_work = cell(K, 1);
Amubeta = [];
temp_sige = [];
InitialTau = cell(K, 1);
ydata = cell(K, 1);
KU = cell(K, 1);
for k = 1 : K
    meanS_work{k} = mean(S_work{k}, 2);
    mubetaS_work{k} = regress(meanS_work{k}, B);
    [simulS_work{k}, TauS_work{k}, KUS_work{k}] = initializing(S_work{k}, t, B, p, sigb0, mubetaS_work{k}, w, v0, sige, nu1, nu2, phi, Distribution1, Distribution2, niter);
    mubetaS_work{k} = median(simulS_work{k}(isample, 1 : p), 1)';
    Amubeta = [Amubeta mubetaS_work{k}];
    temp_sige = [temp_sige; simulS_work{k}(isample, p + 3)];
    sige = median(temp_sige);
    InitialTau{k} = InitialValueTau(TauS_work{k}, nn(k), isample);
    ydata{k} = S_work{k};
    KU{k} = InitialValueKU(KUS_work{k}, nn(k), isample);
end


%% Parameter Estimation

% The first step of parameter estimation : \mu(t)
[FixedPara, Sigb0] = FirstStepParaEst(ydata, InitialTau, Amubeta, t, n_c, sige, p, niter, B, KU);
ycenter = cell(K, 1);
Aycenter = [];
for k = 1 : K
    ycenter{k} = ydata{k} - kron(ones(1, nn(k)), B * mubetaS_work{k});
    Aycenter = [Aycenter ycenter{k}];
end

% The second step of parameter estimation : \tau(t) & \epsilon(t)
[RandomPara, ATau] = SecondStepParaEst(Aycenter, m, n, xi, t, sige, Distribution1, Distribution2, niter, phi, p);


output_train.FixedPara = FixedPara;
output_train.Sigb0 = Sigb0;
output_train.RandomPara = RandomPara;
output_train.ATau = ATau;
output_train.m = m;
output_train.B = B;




