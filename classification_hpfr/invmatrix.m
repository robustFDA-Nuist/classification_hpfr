function [invA, invcovm] = invmatrix(covm, sige)
KU1 = 1;
KU2 = 1;
[U1, S1] = svd(covm);
temp = 1/(sige * KU1) + 1./ (diag(S1) * KU2);
temp = 1./ temp; temp = diag(temp);
invA = U1 * temp *U1';
invA = (invA + (invA)') / 2;

s = 1e-8;
m = size(covm, 1);
[U2, S2] = svd(covm + s * eye(m));
invcovm =  U2 * diag(1./ diag(S2)) * U2';
invcovm = (invcovm + invcovm')/2;
end

