function  [fx, dfx] = exppdc(xi, t, y, U1, U2, sige)
%xi = [logw, logv0 ,loga1, loga0]
expxi = exp(xi);%[exp(xi(1)), exp(xi(2))] = [w, v0]
m = length(t);
n = size(y, 2);
t = t(:); %将时间t转化成列向量

covm = exp(-0.5 * expxi(1) * (t * ones(1, m) - ones(m, 1) * (t)') .^2);
covm = expxi(2) * covm + 1e-1 * eye(m);
% covm = covm + expxi(3) * t * (t)';
% covm = covm + expxi(4);

fx = 0.5 * xi(1) + 0.5 * exp(-xi(1)) + 0.5 * (xi(2) + 1)^2;

dfx(1) = 0.5 - 0.5 * exp(-xi(1));
dfx(2) = 1 + xi(2);
% dfx(3) = 1/9 * (xi(3) + 3);
% dfx(4) = 1/9 * (xi(4) + 3);
[U, S, ~] = svd(covm);
for i = 1 : n
    sig = 1 / U2(i) * covm + sige / U1(i) * eye(m);
    logdetsig = log(abs(det(sig)));
    temp = diag(S) / U2(i) + sige / U1(i);
    temp = 1 ./ temp;
    invsig = U * diag(temp) * U';
%     invsig - inv(sig)
    invsigy = invsig * y(:, i);
    
    fx = fx + 0.5 * logdetsig + 0.5 * y(:, i)' * invsig * y(:, i);
    
    V1 = (1 / U2(i)) * expxi(2) * exp(-0.5 * expxi(1) * (t * ones(1, m) - ones(m, 1) * (t)') .^2)...
        .* ((-0.5 * expxi(1) * (t * ones(1, m) - ones(m, 1) * (t)') .^2));
    dfx(1) = dfx(1) + 0.5 * trace(invsig * V1) - 0.5 * invsigy' * V1 * invsigy;
    
    V2 = (1 / U2(i)) * expxi(2) * exp(-0.5 * expxi(1) * (t * ones(1, m) - ones(m, 1) * (t)') .^2);
    dfx(2) = dfx(2) + 0.5 * trace(invsig * V2) - 0.5 * invsigy' * V2 * invsigy;
    
%     V3 = (1 / U2(i)) * expxi(3) * t * (t)';
%     dfx(3) = dfx(3) + 0.5 * trace(invsig * V3) - 0.5 * invsigy' * V3 * invsigy;
    
%     V4 = (1 / U2(i)) * expxi(4) * ones(m, m);
%     dfx(4) = dfx(4) + 0.5 * trace(invsig * V4) - 0.5 * invsigy' * V4 * invsigy;
%     dfx(4) = dfx(4) + 0.5 *  (1 / nu2) * expxi(4) * trace(invsig) - 0.5 * (1 / nu2) * expxi(4) * (invsigy)' * invsigy;
end

end

