function meanKU = update_value_KU(KU, n, isample)
m = size(KU, 2);
meanKU = zeros(m, n);
for i = 1 : m
    meanKU(i, :) = mean(cell2mat(KU{i}(isample)'), 1);
end

