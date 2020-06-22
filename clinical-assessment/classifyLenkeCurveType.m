function [curveType, curveTypeProbabilities] = classifyLenkeCurveType(cobbAngles)
%CLASSIFYLENKECURVETYPE Summary of this function goes here
%   Detailed explanation goes here
pt = cobbAngles(1);
mt = cobbAngles(2);
tl = cobbAngles(3);

% classify curve type
if (pt < 25) && (mt >= 25) && (tl < 25)
    curveType = 1;
elseif (pt >= 25) && (tl < 25)
    curveType = 2;
elseif (pt < 25) && (mt >= 25) && (tl >= 25) && (mt >= tl)
    curveType = 3;
elseif (pt >= 25) && (tl >= 25)
    curveType = 4;
elseif (pt < 25) && (mt < 25) && (tl >= 25)
    curveType = 5;
elseif (pt < 25) && (mt >= 25) && (tl >= 25) && (mt < tl)
    curveType = 6;
elseif (tl > mt) && (tl > pt)
    curveType = 5;
else
    curveType = 1;
end

%%%%%%%% estimate probability of each curve type
%%%%%%%% assuming estimate lies in the center of normal distribution

sd_est = 6.86; % standard deviation of estimated Cobb angle

p_tl_major = probAGreaterThanB(tl,mt,sd_est)*probAGreaterThanB(tl,pt,sd_est);
p_mt_major = 1 - p_tl_major;

p_curvesLess25 = zeros(1, 3);
count = 1;
for ang = [pt mt tl]
    p_curvesLess25(count) = normcdf(25,ang,sd_est);
    count = count + 1;
end
p_curvesGreater25 = 1 - p_curvesLess25;

curveTypeProbabilities = zeros(1, 6);

curveTypeProbabilities(1) = ((p_curvesLess25(1)) * (p_curvesGreater25(2)) * (p_curvesLess25(3))) + ((p_curvesLess25(1)) * (p_curvesLess25(2)) * (p_curvesLess25(3)) * p_mt_major);
curveTypeProbabilities(2) = ((p_curvesGreater25(1)) * (p_curvesGreater25(2)) * (p_curvesLess25(3))) + ((p_curvesGreater25(1)) * (p_curvesLess25(2)) * (p_curvesLess25(3)));
curveTypeProbabilities(3) = ((p_curvesLess25(1)) * (p_curvesGreater25(2)) * (p_curvesGreater25(3)) * p_mt_major);
curveTypeProbabilities(4) = ((p_curvesGreater25(1)) * (p_curvesGreater25(2)) * (p_curvesGreater25(3))) + ((p_curvesGreater25(1)) * (p_curvesLess25(2)) * (p_curvesGreater25(3)) * p_mt_major);
curveTypeProbabilities(5) = ((p_curvesLess25(1)) * (p_curvesLess25(2)) * (p_curvesGreater25(3))) + ((p_curvesLess25(1)) * (p_curvesLess25(2)) * (p_curvesLess25(3)) * p_tl_major);
curveTypeProbabilities(6) = ((p_curvesLess25(1)) * (p_curvesGreater25(2)) * (p_curvesGreater25(3)) * p_tl_major);

end


function [prob] = probAGreaterThanB(angA, angB, sd_est)
step = 0:0.01:180;
angA_pdf = normpdf(step,angA,sd_est);
angB_pdf = normpdf(step,angB,sd_est);
if angA > angB
    prob = (1-(trapz(min(angA_pdf, angB_pdf))/(2*trapz(angA_pdf))));
else
    prob = (trapz(min(angA_pdf, angB_pdf))/(2*trapz(angA_pdf)));
end
end
