function [endplateLandmarks, centroidsTop2Bottom] = fitEndplates(spineMask)
%CALCULATEENDPLATESLOPES: fit endplates to a set of vertebral
%segmentations.

%imshow(spineMask)
%hold on

endplateLandmarks = [];

% get stats with region props
stats = regionprops(spineMask, 'Centroid');
% find centroids
centroids = cat(1,stats.Centroid);

%%%%%%%%%%%%%%%%% loop through vertebrae from top to bottom
L = bwlabel(spineMask);
[~, indexTopBottom] = sort(centroids(:,2));
centroidsTop2Bottom = centroids(indexTopBottom,:);
indexTopBottom = indexTopBottom';
for n = indexTopBottom
    vertebraMask = ismember(L, n);
    
    %%%%%%%%%%%%%%%%%%%%% find perimeter and fit minimum bounding rectangle
    perims = bwboundaries(vertebraMask);
    perim = perims{1,1};
    %plot(perim(:,2), perim(:,1));
    [rectx,recty,~,~] = minboundrect(perim(:,2), perim(:,1));
    %plot(rectx, recty)
    
    rectx = rectx(1:4);
    recty = recty(1:4);
    
    [~, indexRect] = sort(rectx);
    
    [~, TLind] = min(recty(indexRect(1:2)));
    [~, TRind] = min(recty(indexRect(3:4)));
    
    TLind = indexRect(TLind);
    TRind = indexRect(TRind+2);
    
    %plot(rectx(TLind), recty(TLind), '+');
    %plot(rectx(TRind), recty(TRind), 'o');
    
    %%%%%%%%%%%%%%%%%%%%% rotate by the angle of the fitted rectangle
    angle = rad2deg(atan2(recty(TRind)-recty(TLind), rectx(TRind)-rectx(TLind)));
    vertebraMask = imrotate(vertebraMask,angle);
    
    %%%%%%%%%%%%%%%%%%%%% find left and right edge at the centroid height
    % get stats with region props
    stats = regionprops(vertebraMask, 'Centroid');
    % find centroids
    centroids = cat(1,stats.Centroid);
    yCentroid = centroids(2);
    centroidRow = round(yCentroid);
    % Extract
    oneRow = vertebraMask(centroidRow, :);
    % Get left and right columns.
    leftColumn = find(oneRow, 1, 'first');
    rightColumn = find(oneRow, 1, 'last');
    
    %%%%%%%%%%%%%%%%%%%%%% extract points on the top and bottom endplates
    plateLen = rightColumn-leftColumn;
    leftCol = round(leftColumn+(plateLen/15));
    rightCol = round(rightColumn-(plateLen/15));

    lineLen = rightCol-leftCol;
    lineCols = leftCol:rightCol;
    lineTopRow = zeros(1, lineLen);
    lineBottomRow = lineTopRow;
    for k = lineCols
        idx = k-leftCol+1;
        % Extract
        oneColumn = vertebraMask(:, k);
        % Get top and bottom row
        lineTopRow(idx) = find(oneColumn, 1, 'first');
        lineBottomRow(idx) = find(oneColumn, 1, 'last');
    end
    
    %%%%%%%%%%%%%%%%%%%%% fit lines to endplates
    p = polyfit(lineCols, lineTopRow, 1);
    topLine = polyval(p, lineCols);

    p = polyfit(lineCols, lineBottomRow, 1);
    bottomLine = polyval(p, lineCols);
    
    corners = [lineCols(1) topLine(1);
                lineCols(1) bottomLine(1);
                lineCols(length(lineCols)) topLine(length(lineCols));
                lineCols(length(lineCols)) bottomLine(length(lineCols))]';
    
    % Create rotation matrix
    alpha = angle; % to rotate 90 counterclockwise
    RotMatrix = [cosd(alpha) -sind(alpha); sind(alpha) cosd(alpha)];
    ImCenterA = flipud((size(vertebraMask)/2)');         % Center of the rotated image
    ImCenterB = flipud((size(spineMask)/2)');  % Center of the original image

    corners = round(RotMatrix*(corners-ImCenterA)+ImCenterB);
    %plot(corners(1,[1 3]),corners(2,[1 3]))
    %plot(corners(1,[2 4]),corners(2,[2 4]))
    
    orderedCorners = corners;
    orderedCorners(:,2) = corners(:,3);
    orderedCorners(:,3) = corners(:,2);
    endplateLandmarks = cat(1, endplateLandmarks, orderedCorners');
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Detect errors in endplates

numLandmarks = length(endplateLandmarks);
numVertebrae = numLandmarks/4;
numEndplates = numLandmarks/2;

%%%%%%%%%%%%%%%%%%%% find large changes in slope between successive vertebrae and remove
% averaging the endplates for each vertebra to find average slopes
midVertebraLandmarks = zeros(numEndplates,2);
for k = 1:2:numEndplates
    landmarkPos = (k-1)*2+1;
    midVertebraLandmarks(k,:) = (endplateLandmarks(landmarkPos,:) + endplateLandmarks(landmarkPos+2,:))/2;
    midVertebraLandmarks(k+1,:) = (endplateLandmarks(landmarkPos+1,:) + endplateLandmarks(landmarkPos+3,:))/2;
end
% calculate slope of each vertebra
midSlopes = zeros(numVertebrae,1);
for k = 1:numVertebrae
    landmarkPos = (k-1)*2+1;
    midSlopes(k,1) = (midVertebraLandmarks(landmarkPos+1,2) - midVertebraLandmarks(landmarkPos,2)) / (midVertebraLandmarks(landmarkPos+1,1) - midVertebraLandmarks(landmarkPos,1));
end
% traverse vertebra slopes to find any unusual changes
for k = 1:numVertebrae-2
    % if angle between consecutive vertebrae is greater than 45 degrees
    % assume error and replace with weighted avg of neighbours
    if abs(rad2deg(atan(midSlopes(k+1))-atan(midSlopes(k)))) > 45
        landmarkPos = (k)*4+1; % k instead of k-1 here because vertebra k+1 is unusual slope
        endplateLandmarks(landmarkPos,:) = (3*endplateLandmarks(landmarkPos-2,:) + endplateLandmarks(landmarkPos+4,:)) / 4;
        landmarkPos = (k)*4+2;
        endplateLandmarks(landmarkPos,:) = (3*endplateLandmarks(landmarkPos-2,:) + endplateLandmarks(landmarkPos+4,:)) / 4;
        landmarkPos = (k)*4+3;
        endplateLandmarks(landmarkPos,:) = (endplateLandmarks(landmarkPos-4,:) + 3*endplateLandmarks(landmarkPos+2,:)) / 4;
        landmarkPos = (k)*4+4;
        endplateLandmarks(landmarkPos,:) = (endplateLandmarks(landmarkPos-4,:) + 3*endplateLandmarks(landmarkPos+2,:)) / 4;
        
        
        % recalculate vertebra slopes with changes
        % averaging the endplates for each vertebra to find average slopes
        midVertebraLandmarks = zeros(numEndplates,2);
        for k2 = 1:2:numEndplates
            landmarkPos = (k2-1)*2+1;
            midVertebraLandmarks(k2,:) = (endplateLandmarks(landmarkPos,:) + endplateLandmarks(landmarkPos+2,:))/2;
            midVertebraLandmarks(k2+1,:) = (endplateLandmarks(landmarkPos+1,:) + endplateLandmarks(landmarkPos+3,:))/2;
        end

        % calculate slope of each vertebra
        midSlopes = zeros(numVertebrae,1);
        for k2 = 1:numVertebrae
            landmarkPos = (k2-1)*2+1;
            midSlopes(k2,1) = (midVertebraLandmarks(landmarkPos+1,2) - midVertebraLandmarks(landmarkPos,2)) / (midVertebraLandmarks(landmarkPos+1,1) - midVertebraLandmarks(landmarkPos,1));
        end
    end
end

%%%%%%%%%%%%%%%%%%% calculate length of each endplate and remove if unusual size
midLengths = zeros(numVertebrae,1);
for k = 1:numVertebrae
    landmarkPos = (k-1)*2+1;
    midLengths(k,1) = norm(midVertebraLandmarks(landmarkPos+1,:) - midVertebraLandmarks(landmarkPos,:));
end
rows2Delete = [];
for k = 1:numVertebrae
    if (midLengths(k) < 0.6*median(midLengths)) || (midLengths(k) > 2*median(midLengths))
        landmarkPos = (k-1)*4+1;
        rows2Delete = [rows2Delete landmarkPos landmarkPos+1 landmarkPos+2 landmarkPos+3];
    end
end
endplateLandmarks(rows2Delete,:) = [];

numLandmarks = length(endplateLandmarks);
numVertebrae = numLandmarks/4;
numEndplates = numLandmarks/2;

%%%%%%%%%%%%%%%%%%% find large changes in slope between endplates of each vertebra and remove
% calculate slope of each vertebra
endplateSlopes = zeros(numEndplates,1);
for k = 1:numEndplates
    landmarkPos = (k-1)*2+1;
    endplateSlopes(k,1) = (endplateLandmarks(landmarkPos+1,2) - endplateLandmarks(landmarkPos,2)) / (endplateLandmarks(landmarkPos+1,1) - endplateLandmarks(landmarkPos,1));
end
% for T1 and L5, if angle between endplates is too big, replace most sloped
% with same slope of the other endplate
for k = [1 numVertebrae]
    endplatePos = (k-1)*2+1;
    % if angle between endplates on a vertebrae is greater than 10 degrees
    if abs(rad2deg(atan(endplateSlopes(endplatePos+1))-atan(endplateSlopes(endplatePos)))) > 10
        if abs(endplateSlopes(endplatePos)) > abs(endplateSlopes(endplatePos+1))
            landmarkPos = (k-1)*4+1;
            endplateLandmarks(landmarkPos,2) = endplateLandmarks(landmarkPos+1,2) + (endplateLandmarks(landmarkPos+2,2)-endplateLandmarks(landmarkPos+3,2));
        else
            landmarkPos = (k-1)*4+3;
            endplateLandmarks(landmarkPos,2) = endplateLandmarks(landmarkPos+1,2) + (endplateLandmarks(landmarkPos-2,2)-endplateLandmarks(landmarkPos-1,2));
        end
    end
end
% for the rest ...
for k = 2:numVertebrae-1
    endplatePos = (k-1)*2+1;
    % if angle between endplates on a vertebrae is greater than 10 degrees
    if abs(rad2deg(atan(endplateSlopes(endplatePos+1))-atan(endplateSlopes(endplatePos)))) > 10
        % find which endplate is most deviated from its neighbouring
        % endplate on the adjacent vertebra; and replace the most deviated
        % endplate with the weighted avg of its 2 neighbours
        if abs(rad2deg(atan(endplateSlopes(endplatePos))-atan(endplateSlopes(endplatePos-1)))) > abs(rad2deg(atan(endplateSlopes(endplatePos+2))-atan(endplateSlopes(endplatePos+1))))
            landmarkPos = (k-1)*4+1;
            endplateLandmarks(landmarkPos,:) = (3*endplateLandmarks(landmarkPos-2,:) + endplateLandmarks(landmarkPos+2,:)) / 4;
            landmarkPos = (k-1)*4+2;
            endplateLandmarks(landmarkPos,:) = (3*endplateLandmarks(landmarkPos-2,:) + endplateLandmarks(landmarkPos+2,:)) / 4;
        else
            landmarkPos = (k-1)*4+3;
            endplateLandmarks(landmarkPos,:) = (endplateLandmarks(landmarkPos-2,:) + 3*endplateLandmarks(landmarkPos+2,:)) / 4;
            landmarkPos = (k-1)*4+4;
            endplateLandmarks(landmarkPos,:) = (endplateLandmarks(landmarkPos-2,:) + 3*endplateLandmarks(landmarkPos+2,:)) / 4;
        end
    end
end

end

