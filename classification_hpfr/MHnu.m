function last1 = MHnu(last,U,lambda,prior, sigma)
% This is the main function for MH algorithm for \nu.


if (prior == 'Hierar')

    % The following is to generate truncated normal distribution of random numbers!!
    cand = tnormrnd(3, 10 , last, sigma, 1, 1);

    [ff1] = gHierar(cand,U,lambda);
    [ff2] = gHierar(last,U,lambda);

    drtunclast = dtruncnorm(last, 3, 10, cand, sigma);
    drtunccand =  dtruncnorm(cand, 3, 10, last, sigma);

    alfa = (exp(ff1)/exp(ff2)) * (drtunclast/drtunccand);
end
if (unifrnd(0,1,1) < min(alfa,1))
    last1 = cand;
else
    last1 = last;
end
end

