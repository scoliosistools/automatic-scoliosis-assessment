function [output] = inLimits(desiredVal,maxSize)
%INLIMITS Summary of this function goes here
%   Detailed explanation goes here
if desiredVal < 1
    output = 1;
elseif desiredVal > maxSize
    output = maxSize;
else
    output = desiredVal;
end
end

