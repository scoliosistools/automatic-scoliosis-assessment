function [cobbAngles, apicalVertebrae, angleLocations, cobbEndplates] = calculateCobbAngles(endplateLandmarks, centroids, angleLocations)
%CALCULATECOBBANGLES: calculate the PT, MT, and TL/L Cobb angles from a
%given set of endplate landmarks.

numLandmarks = length(endplateLandmarks);
numVertebrae = numLandmarks/4;
numEndplates = numLandmarks/2;

cobbAngles = zeros(3,1);
apicalVertebrae = zeros(3,1);
cobbEndplates = zeros(4,2,3);

%%%%%%%%%%%%%%%%% averaging the endplates for each vertebra to find average slopes
midVertebraLandmarks = zeros(numEndplates,2);
for k = 1:2:numEndplates
    landmarkPos = (k-1)*2+1;
    midVertebraLandmarks(k,:) = (endplateLandmarks(landmarkPos,:) + endplateLandmarks(landmarkPos+2,:))/2;
    midVertebraLandmarks(k+1,:) = (endplateLandmarks(landmarkPos+1,:) + endplateLandmarks(landmarkPos+3,:))/2;
end

%%%%%%%%%%%%%%%%% calculate slope of each vertebra
midSlopes = zeros(numVertebrae,1);
for k = 1:numVertebrae
    landmarkPos = (k-1)*2+1;
    midSlopes(k,1) = (midVertebraLandmarks(landmarkPos+1,2) - midVertebraLandmarks(landmarkPos,2)) / (midVertebraLandmarks(landmarkPos+1,1) - midVertebraLandmarks(landmarkPos,1));
end

if angleLocations == zeros(4,1) % can manually set angle locations either
    %%%%%%%%%%%%%%%%% find mt apex using centroids
    [~, indexTopBottom] = sort(centroids(:,2));
    thoracicCentroids = centroids(indexTopBottom(6:11),1) - mean(centroids(1:11,1));

    [maxima, maximaInd] = findpeaks(thoracicCentroids);
    [~, maxPkInd] = max(maxima);

    thoracicCentroidsInv = max(thoracicCentroids) - thoracicCentroids;
    [minima, minimaInd] = findpeaks(thoracicCentroidsInv);
    [~, minPkInd] = max(minima);

    pks = [maximaInd(maxPkInd) minimaInd(minPkInd)];

    [~, apexInd] = max(abs(thoracicCentroids(pks)));

    if ~isempty(apexInd)
        apicalVertebrae(2) = pks(apexInd)+5;
    else
        apicalVertebrae(2) = 9;
    end

    %%%%%%%%%%%%%%%%% find most tilted vertebra for each angle
    [~, ind] = max(abs(midSlopes(4:apicalVertebrae(2)-1)));
    angleLocations(2) = ind(1)+3;

    [~, ind] = max(abs(midSlopes(1:angleLocations(2))-midSlopes(angleLocations(2))));
    angleLocations(1) = ind(1);

    if numVertebrae >= 14
        [~, ind] = max(abs(midSlopes((apicalVertebrae(2)+1):14)-midSlopes(angleLocations(2))));
    else
        [~, ind] = max(abs(midSlopes((apicalVertebrae(2)+1):numVertebrae)-midSlopes(angleLocations(2))));
    end
    angleLocations(3) = ind(1)+(apicalVertebrae(2));

    [~, ind] = max(abs(midSlopes(angleLocations(3):numVertebrae)-midSlopes(angleLocations(3))));
    angleLocations(4) = ind(1)+(angleLocations(3)-1);

    %%%%%%%%%%%%%%%%% find pt and tl/l apical vertebrae using slopes
    apex1vector = midSlopes(angleLocations(1)+1:angleLocations(2)-1) - ((midSlopes(angleLocations(1))+midSlopes(angleLocations(2)))/2);
    [~, ind] = min(abs(apex1vector));
    if ~isempty(ind)
        apicalVertebrae(1) = ind(1)+angleLocations(1);
    end

    apex2vector = midSlopes(angleLocations(3)+1:angleLocations(4)-1) - ((midSlopes(angleLocations(3))+midSlopes(angleLocations(4)))/2);
    [~, ind] = min(abs(apex2vector));
    if ~isempty(ind)
        apicalVertebrae(3) = ind(1)+angleLocations(3);
    end
end

%%%%%%%%%%%%%%%%% calculate superior and inferior endplate slopes
superiorLandmarks = zeros(numEndplates,2);
inferiorLandmarks = zeros(numEndplates,2);
for k = 1:2:numEndplates
    landmarkPos = (k-1)*2+1;
    superiorLandmarks(k,:) = endplateLandmarks(landmarkPos,:);
    superiorLandmarks(k+1,:) = endplateLandmarks(landmarkPos+1,:);
    inferiorLandmarks(k,:) = endplateLandmarks(landmarkPos+2,:);
    inferiorLandmarks(k+1,:) = endplateLandmarks(landmarkPos+3,:);
end

superiorSlopes = zeros(numVertebrae,1);
inferiorSlopes = zeros(numVertebrae,1);
for k = 1:numVertebrae
    landmarkPos = (k-1)*2+1;
    superiorSlopes(k,1) = (superiorLandmarks(landmarkPos+1,2) - superiorLandmarks(landmarkPos,2)) / (superiorLandmarks(landmarkPos+1,1) - superiorLandmarks(landmarkPos,1));
    inferiorSlopes(k,1) = (inferiorLandmarks(landmarkPos+1,2) - inferiorLandmarks(landmarkPos,2)) / (inferiorLandmarks(landmarkPos+1,1) - inferiorLandmarks(landmarkPos,1));
end

%%%%%%%%%%%%%%%%% calculate Cobb angles
cobbAngles(1) = abs(rad2deg(atan(superiorSlopes(angleLocations(1)))-atan(inferiorSlopes(angleLocations(2)))));
cobbAngles(2) = abs(rad2deg(atan(superiorSlopes(angleLocations(2)))-atan(inferiorSlopes(angleLocations(3)))));
cobbAngles(3) = abs(rad2deg(atan(superiorSlopes(angleLocations(3)))-atan(inferiorSlopes(angleLocations(4)))));

%%%%%%%%%%%%%%%%% return endplates used in calculation for plotting
for k = 1:3
    landmarkPos = (angleLocations(k)-1)*2+1;
    cobbEndplates(1:2,:,k) = superiorLandmarks(landmarkPos:landmarkPos+1,:);
    landmarkPos = (angleLocations(k+1)-1)*2+1;
    cobbEndplates(3:4,:,k) = inferiorLandmarks(landmarkPos:landmarkPos+1,:);
end

end

