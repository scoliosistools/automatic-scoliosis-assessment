function [cobbAngles,outputOverlay] = predictedSpineMask2Outputs(xray,spineMask)
%PREDICTEDSPINEMASK2OUTPUTS interface with clinical assessment algorithms
%to provide relevant metrics given an inputted predictied vertebral segmentation map

%% call algorithms to perform assessment
xray_size = size(xray);
spineMask = imresize(spineMask, xray_size);
spineMask = processSpineMask(spineMask);

[endplateLandmarks, centroidsTop2Bottom] = fitEndplates(spineMask);

nullAngleLocations = zeros(4,1); % used to allow for manual setting of angle locations
[cobbAngles, ~, ~, cobbEndplates] = calculateCobbAngles(endplateLandmarks, centroidsTop2Bottom, nullAngleLocations);

%% create rgb image to show algorithm process
outputOverlay = zeros([xray_size 3]);
outputOverlayR = xray;
outputOverlayG = xray;
outputOverlayB = xray;

% segmentation overlay
spineMaskPerim = imdilate(bwperim(spineMask), strel('disk',5));
outputOverlayR(spineMaskPerim) = 255;
outputOverlayG(spineMaskPerim) = 255;
outputOverlayB(spineMaskPerim) = 255;

% cobb overlay
cobbOverlay = zeros([xray_size 3]);

cobbEndplatesExtended = cobbEndplates;
for n = 1:3 % pt - mt - tl/l
    for m = [0 2]
        cobbEndplatesExtended(1+m,1,n) = cobbEndplates(1+m,1,n)-(cobbEndplates(2+m,1,n)-cobbEndplates(1+m,1,n));
        cobbEndplatesExtended(2+m,1,n) = cobbEndplates(2+m,1,n)-(cobbEndplates(1+m,1,n)-cobbEndplates(2+m,1,n));
        cobbEndplatesExtended(1+m,2,n) = cobbEndplates(1+m,2,n)-(cobbEndplates(2+m,2,n)-cobbEndplates(1+m,2,n));
        cobbEndplatesExtended(2+m,2,n) = cobbEndplates(2+m,2,n)-(cobbEndplates(1+m,2,n)-cobbEndplates(2+m,2,n));
    end
end

for n = 1:3 % pt - mt - tl/l
    for m = [0 2]
        hor = abs(cobbEndplatesExtended(1+m,1,n) - cobbEndplatesExtended(2+m,1,n));
        vert = abs(cobbEndplatesExtended(1+m,2,n) - cobbEndplatesExtended(2+m,2,n));
        line_length = max(hor,vert);
        if cobbEndplatesExtended(1+m,1,n) ~= cobbEndplatesExtended(2+m,1,n)
            lineCols = round(cobbEndplatesExtended(1+m,1,n):(cobbEndplatesExtended(2+m,1,n)-cobbEndplatesExtended(1+m,1,n))/line_length:cobbEndplatesExtended(2+m,1,n));
        else
            lineCols = zeros(1, line_length+1) + cobbEndplatesExtended(1+m,1,n);
        end
        if cobbEndplatesExtended(1+m,2,n) ~= cobbEndplatesExtended(2+m,2,n)
            lineRows = round(cobbEndplatesExtended(1+m,2,n):(cobbEndplatesExtended(2+m,2,n)-cobbEndplatesExtended(1+m,2,n))/line_length:cobbEndplatesExtended(2+m,2,n));
        else
            lineRows = zeros(1, line_length+1) + cobbEndplatesExtended(1+m,2,n);
        end
        for k = 1:line_length+1
            if lineRows(k) < xray_size(1) && lineCols(k) < xray_size(2)
                cobbOverlay(lineRows(k),lineCols(k),n) = 1;
            end
        end
    end
end
ptOverlay = imdilate(cobbOverlay(:,:,1), strel('disk',5));
mtOverlay = imdilate(cobbOverlay(:,:,2), strel('disk',5));
tlOverlay = imdilate(cobbOverlay(:,:,3), strel('disk',5));

outputOverlayR(ptOverlay == 1) = 255;
outputOverlayG(ptOverlay == 1) = 0;
outputOverlayB(ptOverlay == 1) = 0;

outputOverlayR(mtOverlay == 1) = 0;
outputOverlayG(mtOverlay == 1) = 255;
outputOverlayB(mtOverlay == 1) = 0;

outputOverlayR(tlOverlay == 1) = 0;
outputOverlayG(tlOverlay == 1) = 0;
outputOverlayB(tlOverlay == 1) = 255;

outputOverlay(:,:,1) = outputOverlayR;
outputOverlay(:,:,2) = outputOverlayG;
outputOverlay(:,:,3) = outputOverlayB;

outputOverlay = outputOverlay/255;
end

