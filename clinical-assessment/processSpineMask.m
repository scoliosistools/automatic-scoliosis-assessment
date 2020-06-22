function [spineMask] = processSpineMask(spineMask)
%SPINEMASKPROCESSING Summary of this function goes here
%   Detailed explanation goes here

spineMask = imbinarize(spineMask(:,:,1));
numPixels = size(spineMask);
numPixels = numPixels(1)*numPixels(2);

%%%%%%%%%%%%%%%%%%% remove any tiny objects in image
spineMask = bwareafilt(spineMask, [round(numPixels*0.001) numPixels]);

%%%%%%%%%%%%%%%%%%% imerode -> watershed -> dilate horizontally to fill watershed holes
SE = strel('rectangle',[3 1]); %%%%%%%%%% does this lessen slopes?
spineMask = imerode(spineMask,SE);

D = -bwdist(~spineMask, 'cityblock');
L = watershed(D);
spineMask(L == 0) = 0;

SE = strel('rectangle',[1 2]);
spineMask = imdilate(spineMask,SE);

%%%%%%%%%%%%%%%%%%% imerode -> imfill any remaining holes
SE = strel('rectangle',[2 1]);
spineMask = imerode(spineMask,SE);

spineMask = imfill(spineMask,'holes');


%%%%%%%%%%%%%%%%%%% find any objects with outlier area i.e. vertebrae that are still joined
%%%%%%%%%%%%%%%%%%% repeat watershed process for the large outliers found
stats = regionprops(spineMask, 'Area');
objectAreas = cat(1,stats.Area);
TF = isoutlier(objectAreas,'movmedian',10);
% ignore small outliers in this step
for k = 1:length(TF)
    if TF(k)
        if objectAreas(k) < median(objectAreas)
            TF(k) = 0;
        end
    end
end

stopCount = 0;
while (sum(TF) > 0) && (stopCount < 3)
    
    labelMat = bwlabel(spineMask);
    for k = 1:length(TF)
        if TF(k)
            mergedVertebraeMask = ismember(labelMat, k);
            labelMat(labelMat==k) = 0;

            SE = strel('diamond',5);
            mergedVertebraeMask = imerode(mergedVertebraeMask,SE);

            D = -bwdist(~mergedVertebraeMask, 'euclidean');
            L = watershed(D);
            mergedVertebraeMask(L == 0) = 0;

            SE = strel('square',3);
            mergedVertebraeMask = imdilate(mergedVertebraeMask,SE);

            mergedVertebraeMask = imfill(mergedVertebraeMask,'holes');

            labelMat(mergedVertebraeMask==1) = k;

        end
    end

    spineMask = imbinarize(labelMat);
    
    % get stats with region props
    stats = regionprops(spineMask, 'Area');
    objectAreas = cat(1,stats.Area);
    TF = isoutlier(objectAreas,'movmedian',10);
    % ignore small outliers in this step
    for k = 1:length(TF)
        if TF(k)
            if objectAreas(k) < median(objectAreas)
                TF(k) = 0;
            end
        end
    end
    
    stopCount = stopCount + 1;
end

%%%%%%%%%%%%%%%%%%% remove small outliers
stats = regionprops(spineMask, 'Area');
objectAreas = cat(1,stats.Area);
labelMat = bwlabel(spineMask);
for k = 1:length(objectAreas)
    if objectAreas(k) < 0.4*median(objectAreas)
        labelMat(labelMat==k) = 0;
    end
end
spineMask = imbinarize(labelMat);


%%%%%%%%%%%%%%%%%%%%%% remove outliers too far from spinal column
flag = true;
while flag
    labelMat = bwlabel(spineMask);
    stats = regionprops(spineMask, 'Centroid', 'Area');
    objectAreas = cat(1,stats.Area);
    objectCentroids = cat(1,stats.Centroid);
    numObjects = length(objectCentroids);
    [~, sortedCentroidsIndex] = sort(objectCentroids(:,2));
    
    % calculate distance from expected horizontal coordinate for each centroid
    dist2expectedX = zeros(numObjects,1);
    dist2expectedX(1) = abs(objectCentroids(sortedCentroidsIndex(1),1) - objectCentroids(sortedCentroidsIndex(2),1));
    for k = 2:(numObjects-1)
        neighbouringAvgX = (objectCentroids(sortedCentroidsIndex(k+1),1) + objectCentroids(sortedCentroidsIndex(k-1),1))/2;
        dist2expectedX(k) = abs(objectCentroids(sortedCentroidsIndex(k),1) - neighbouringAvgX);
        
    end
    dist2expectedX(numObjects) = abs(objectCentroids(sortedCentroidsIndex(numObjects),1) - objectCentroids(sortedCentroidsIndex(numObjects-1),1));
    
    % if there appears to be an outlier, remove the largest and repeat process
    if ~isempty(find(dist2expectedX > sqrt(mean(objectAreas)), 1))
        [~, maxDistInd] = max(dist2expectedX);
        maxDistInd = maxDistInd(1);
        labelMat(labelMat==sortedCentroidsIndex(maxDistInd)) = 0;
    else
        flag = false;
    end
    spineMask = imbinarize(labelMat);
end

end
