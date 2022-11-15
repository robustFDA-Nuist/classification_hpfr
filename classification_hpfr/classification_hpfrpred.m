function output_pred = classification_hpfrpred(output_train, input_test, time, isample, distribution1, distribution2)
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
%
% output_train:
% input_test:     all the testing data，including: testdata m*n_test matrix,
%                 label n_test * 1(only for compute the ccr);
% time:           The length of observations of the test data (the value of time*m must be an integer);
% isample:        sampling points;
% distribution1:  = 'N'or 'T'or 'S', the process of \epsilon(t));
% distribution2:  = 'N'or 'T'or 'S', the process of \tau(t).

% ==========================================================================




%% Input
FixedPara = output_train.FixedPara;
RandomPara = output_train.RandomPara;
m = output_train.m;
B = output_train.B;

testdata = input_test.testdata;

nisample = length(isample);
label = testdata.label;
nlabel = length(unique(label));
test_data = testdata.testdata;

ntest = zeros(nlabel, 1);
for k = 1 : nlabel
    ntest(k) = length(find(label == k));
end
n_test = sum(ntest); % The number of test subjects

tmin = 0;
tmax = 1;
t = linspace(tmin, tmax, m);

%% --------Classification and Prediction--------
new_test = 1 : n_test;
c = 0; RMSE_pre = zeros(n_test, 1);
ob_test = 1 : time * m; ob_pre = time * m + 1 : m; %The observations of a new test subject.

for ii = 1 : n_test
    num_test = new_test(ii); % the num_test-th test subject 
    testdata = test_data(:, num_test);

    nob_pre = length(ob_pre);
    fc = zeros(nlabel, 1);
    postpredprok = zeros(nlabel, 1);
    odd = zeros(nlabel, 1);
    Expectation_S = cell(nlabel, 1);
    KU1_S = cell(nlabel, 1);
    KU2_S = cell(nlabel, 1);
    E_S = zeros(nlabel, nob_pre);
    for k = 1 : nlabel
        fc(k) = ntest(k) ./ n_test;
        %--------------
        % Classification
        % Computation of posterior predictive probability for each class
        %--------------
        postpredprok(k) = postpredpro(testdata(ob_test), RandomPara(:, 3), RandomPara(:, [1, 2]), B, FixedPara, isample, k, t, ob_test);
        odd(k) = (postpredprok(k) / postpredprok(1)) * (fc(k) / fc(1));
        %--------------
        % Prediction
        % The weighted prediction based on posterior predictive probability
        %--------------
        Expectation_S{k} = cell(nisample, 1);
        nn = 1;
        for i = isample
            covm = kernelfun(log(RandomPara(i, [1, 2])), ob_test);
            [invA, invcovm] = invmatrix(covm, RandomPara(i, 3));
            for j = 1 : nn
                % Computation of predictions in each class
                [KU1_S{k}(i, j), KU2_S{k}(i, j)] = LatentVariable(t(ob_test), distribution1, distribution2, testdata(ob_test) ,B(ob_test, :), FixedPara{k}(:, i), RandomPara(i, 3), invA, invcovm);
                Expectation_S{k}{i}(:, j) = prediction_label_k(testdata(ob_test), RandomPara(i, [1, 2]), t, t(ob_test), t(ob_pre), (FixedPara{k}(:, i))', B(ob_test, :), B(ob_pre, :), KU1_S{k}(i, j), KU2_S{k}(i, j), RandomPara(i, 3));
            end
        end
        E_S(k, :) = mean(cell2mat(Expectation_S{k}'), 2);
    end
    %--------------
    % ccr
    %--------------
    Pr(ii, :) = odd ./ sum(odd);
    % Find the max value of posterior probability
    [~, position] = max(Pr(ii, :));
    if label(ii) == position
        c = c + 1;
    end
    %--------------
    % Weighted predictions
    %--------------
    for k = 1 : nlabel
        E = zeros(1, length(ob_pre));
        E = E + Pr(ii, k) * E_S(k, :);
    end
    RMSE_pre(ii) = sqrt(mean((E' - testdata(ob_pre)).^2));
end
mean_Rmse_pre = mean(RMSE_pre);
c = c/ n_test;

output_pred.rmse_pred = mean_Rmse_pre;
output_pred.ccr = c;