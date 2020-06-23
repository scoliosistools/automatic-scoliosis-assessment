function [predSlopesAligned] = alignSlopeVectors(gtSlopes,predSlopes)
%ALIGNSLOPEVECTORS: function designed to align the predicted and
%ground-truth endplate slopes for comparison, for cases where there are a 
%different number of predicted endplates.

gtLength = length(gtSlopes);
diffLength = length(predSlopes) - length(gtSlopes);
if diffLength > 0
    for k = 1:diffLength+1
        error = sum(abs(gtSlopes - predSlopes(k:(k+gtLength-1))));
        if (k == 1) || (error < minError)
            minError = error;
            predSlopesAligned = predSlopes(k:(k+gtLength-1));
        end
    end
elseif diffLength < 0
    for k = 1:abs(diffLength)+1
        error = sum(abs(gtSlopes(k:(k+length(predSlopes)-1)) - predSlopes));
        if (k == 1) || (error < minError)
            minError = error;
            predSlopesAligned = zeros(1,gtLength);
            predSlopesAligned(k:(k+length(predSlopes)-1)) = predSlopes;
            predSlopesAligned(1:k) = predSlopes(1);
            predSlopesAligned((k+length(predSlopes)-1):gtLength) = predSlopes(length(predSlopes));
        end
    end
else
    predSlopesAligned = predSlopes;
end

end

