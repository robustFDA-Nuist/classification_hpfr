function [meanInitialValueTau] = update_value_tau_k(Tau, n, isample)
j = 1;
for i = isample
    tau{j} = Tau{i};
    j = j + 1;
end
InitialValueTau1 = [];
for i = 1 : n
    for j = 1 : length(tau)
        InitialValueTau1 = [InitialValueTau1, tau{j}(:, i)];
    end
    meanInitialValueTau(:, i) = mean(InitialValueTau1, 2);
    InitialValueTau1 = [];
end
end

