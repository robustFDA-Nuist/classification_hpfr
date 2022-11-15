clc
clear 
close all
warning off
addpath('fdaM\');
addpath('classification_hpfr\');
tic

%----------------
% Load Data
%----------------
load traindata.mat %load training data
load testdata.mat %load testing data

% The number of orders and breaks of basis functions
norder = 14;
nbreaks = 2;

niter = 4000; %The number of samples
burnin = 2000; % The first burnin samples of burn-in period
thin = 10; 
isample = (burnin+1) : thin : niter;

Distribution1 = 'N'; % The process of \epsilon(t)
Distribution2 = 'N'; % The process of \tau(t)

%-------- Training Data ------
input.traindata  = traindata;

%-------- Testing Data ------
input_test.testdata = testdata;

%--------------
% Training ....
%----------------
output_train = classification_hpfrtrain(input, Distribution1, Distribution2, nbreaks, norder, isample, niter);

%--------------
% Predicting ....
%----------------
time = 1/2; % The length of observations of the test data
output_pred = classification_hpfrpred(output_train, input_test, time, isample, Distribution1, Distribution2);
mean_Rmse_pre = output_pred.rmse_pred; % Mean RMSE of the test data in prediction
ccr = output_pred.ccr; % Correct rate of the test data in classification

toc




