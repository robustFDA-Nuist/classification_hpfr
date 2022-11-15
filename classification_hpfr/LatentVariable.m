function [KU1, KU2] = LatentVariable(t, distribution1, distribution2, y ,B, mubeta, sige, invA, invcovm)
KU1 = 1;
KU2 = 1;
m = length(t);
p = 10;
a = (1 / (KU1 * sige)) * (y - B * mubeta);
tau = (mvnrnd(invA * a, invA, 1))';
    if distribution1 == 'N'
        KU1 = 1;
    elseif distribution1 == 'T'
        nu1 = 4;
        bt1 = nu1/2 + 1 / (2*sige) * (y- B * mubeta - tau)' * (y - B*mubeta - tau);
        U1 = gamrnd((nu1 + m)/2, 1/bt1, 1);
        KU1 = 1 / U1;
    elseif distribution1 == 'S'
        nu1 = 1.6;       
        bt1 = 1 / (2*sige) * (y - B * mubeta - tau)' * (y - B*mubeta - tau);
        U1 = tgamrnd(0, 1, nu1 + m/2, 1/bt1, 1);
        KU1 = 1 / U1;
    end

    if distribution2 == 'N'
        KU2 = 1;
    elseif distribution2 == 'T'
        nu2 = 4;
        bt2 = nu2/2 + tau' * invcovm *tau;
        U2 = gamrnd((nu2 + p)/2, 1/bt2, 1);%V2
        KU2 = 1 / U2;
    elseif distribution2 == 'S'
        nu2 = 1.6;
        bt2 = 0.5 * tau' * invcovm* tau;
        U2 = tgamrnd(0, 1, nu2 + p/2, 1/bt2, 1);
        KU2 = 1 / U2;
    end
end

