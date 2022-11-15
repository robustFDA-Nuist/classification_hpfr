function Expectation = prediction_label_k(y, theta, t, t1, t2, mubeta, B1, B2, KU1, KU2, sige)

m = length(t);
m1 = length(t1);
m2 = length(t2);
covm = kernelfun(log(theta), t);
covm = KU2 * covm + KU1 * sige * eye(m);
Sig = mat2cell(covm, [m1, m2], [m1, m2]);
sig11 = Sig{1, 1};
sig21 = Sig{2, 1};
% s = 1e-8;
s = 0;
[U, S] = svd(sig11 + s * eye(m1));
invsig11 =  U * diag(1./ diag(S)) * U';
invsig11 = (invsig11 + invsig11')/2;
Expectation = B2 * mubeta' + sig21 * invsig11 * (y - B1 * mubeta');





