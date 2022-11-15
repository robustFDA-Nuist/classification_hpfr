function output_pred = predicting(output_train, input_test, time, isample, nisample, Distribution1, Distribution2)
% ============
% Description:
% ============
% This is the main function for classification and weighted prediction.
% Includes the following steps:
%
% 1) Classification :
%    Computation of posterior predictive probability for each label;
%
% 2) Prediction :
%    Computation of predictions in each class and computation of weighted
%    predictions based on posterior predictive probabilities.
%
% ============
% Input:
% ============
% output_train:    
% input_test:     all the testing data，including: testdata m*n_test matrix, 
%                 label n_test * 1(only for compute the ccr);
% time:           The length of observations of the test data;
% isample:        sampling points;
% nisample:       number of 
% Distribution1: 
% Distribution2:

%% Input
FixedPara = output_train.FixedPara;
RandomPara = output_train.RandomPara;
m = output_train.m;
B = output_train.B;

nlabel = length(input_test.S_test);
label = input_test.label;
S_test = [];
n_test = 0;
for k = 1 : nlabel
    S_test = [S_test input_test.S_test{k}];
    ntest(k) = size(input_test.S_test{k}, 2);
    n_test = n_test + ntest(k);
end

tmin = 0;
tmax = 1;
t = linspace(tmin, tmax, m);
num_new = n_test; % The number of test subjects

new_test = 1 : num_new;
c = 0; RMSE_pre = zeros(num_new, 1);
ob_test = 1 : time * m; ob_pre = time * m + 1 : m; %The observations of a new test subject.

for ii = 1 : num_new
    num_test = new_test(ii);
    Stest = S_test(:, num_test);
    for k = 1 : nlabel
        fc(k) = ntest(k) ./ n_test;
        % Classification
        % Computation of posterior predictive probability for each class

        postpredprok(k) = postpredpro(Stest(ob_test), RandomPara(:, 3), RandomPara(:, [1, 2]), B, FixedPara, isample, k, t, ob_test);
        odd(k) = (postpredprok(k) / postpredprok(1)) * (fc(k) / fc(1));

        % Prediction
        % The weighted prediction based on posterior predictive probability
        Expectation_S{k} = cell(nisample, 1);
        nn = 1;
        for i = isample
            covm = kernelfun(log(RandomPara(i, [1, 2])), ob_test);
            [invA, invcovm] = invmatrix(covm, RandomPara(i, 3));
            for j = 1 : nn
                % Computation of predictions in each class
                [KU1_S{k}(i, j), KU2_S{k}(i, j)] = LatentVariable(t(ob_test), Distribution1, Distribution2, Stest(ob_test) ,B(ob_test, :), FixedPara{k}(:, i), RandomPara(i, 3), invA, invcovm);
                Expectation_S{k}{i}(:, j) = prediction_label_k(Stest(ob_test), RandomPara(i, [1, 2]), t, t(ob_test), t(ob_pre), (FixedPara{k}(:, i))', B(ob_test, :), B(ob_pre, :), KU1_S{k}(i, j), KU2_S{k}(i, j), RandomPara(i, 3));
            end
        end
        E_S(k, :) = mean(cell2mat(Expectation_S{k}'), 2);
    end
    % ccr
    Pr = odd ./ sum(odd);
    % Find the max value of posterior probability
    [~, position] = max(Pr);
    if label(ii) == position
        c = c + 1;
    end
    % Weighted predictions
    for k = 1 : nlabel
        E = zeros(1, length(ob_pre));
        E = E + Pr(k) * E_S(k, :);
    end
    RMSE_pre(ii) = sqrt(mean((E' - Stest(ob_pre)).^2));
end
mean_Rmse_pre = mean(RMSE_pre);
c = c/ num_new;

output_pred.rmse_pred = mean_Rmse_pre;
output_pred.ccr = c;