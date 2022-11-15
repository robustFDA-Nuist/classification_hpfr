function [ff] = gHierar(nu,U,lambda)
n = length(U);
ff = (-lambda*nu) + 0.5*n*nu*log(nu/2) + (0.5*nu)*sum(log(U) - U) - n*log(gamma(nu/2));

end

