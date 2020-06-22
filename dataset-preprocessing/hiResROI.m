% converting landmarks to high resolution segmentations (ROI - region of interest)
% first for the training dataset and then the testing dataset
for DatabaseToProcess = 0:1
    if ~DatabaseToProcess % training dataset
        % directory to save masks
        roidir = '..\data\HiResVertebraeMasks\';
        % directory of images
        imdir = "..\data\boostnet_labeldata\data\training";
        % read landmarks
        load("..\data\FixedSpineWebData\fixedTrainingLandmarks.mat");
        % read spreadsheet of filenames
        filenames = readmatrix('..\data\boostnet_labeldata\labels\training\filenames.csv', 'ExpectedNumVariables', 1, 'OutputType', 'string', 'Delimiter',',');
    else % testing dataset
        roidir = '..\data\PredictionsVsGroundTruth\SpineMasks_GroundTruthEndplates\';
        imdir = "..\data\boostnet_labeldata\data\test";
        load("..\data\FixedSpineWebData\fixedTestingLandmarks.mat");
        filenames = readmatrix('..\data\boostnet_labeldata\labels\test\filenames.csv', 'ExpectedNumVariables', 1, 'OutputType', 'string', 'Delimiter',',');
    end


    files = dir(fullfile(imdir, '*.jpg'));

    numImages = size(filenames);
    numImages = numImages(1);

    n = 4;  % order of polynomial to fit

    % scale the search zone
    spacing_multiplier = 0.5;
    length_multiplier = 0.5;
    width_multiplier_small = 0.1;
    width_multiplier_big = 0.15;

    % plot to test outputs (change loop to 1:50:numImages and uncomment lines for plotting)
    %figure;
    %plotcount = 1;

    % loop through images
    for m = 1:numImages

        % read through files in correct order by checking filename
        for filecounter =  1:length(files)
            if files(filecounter).name == filenames(m)
                image = imread(fullfile(imdir, files(filecounter).name));
            end
        end

        imSize = size(image);
        threshold = imSize(1)*imSize(2); % 1 -> default, big -> no correction ... imSize(1)*imSize(2)/75000 for some correction

        % initialise empty mask
        BW = false(imSize(1),imSize(2));

        % get landmarks for this image and scale to pixel value
        labels = zeros(68,2);
        labels(:,:) = landmarks(m,:,:);
        labels(:,1) = labels(:,1)*imSize(2);
        labels(:,2) = labels(:,2)*imSize(1);

        %{
        % add C7 and S1 - these are not marked in the spineweb dataset, so we
        % can estimate values using the closest landmarks
        T1Length = mean([(labels(3, 2)-labels(1, 2)) (labels(4, 2)-labels(2, 2))]);
        T1Spacing = mean([(labels(5, 2)-labels(3, 2)) (labels(6, 2)-labels(4, 2))]);
        L5Length = mean([(labels(67, 2)-labels(65, 2)) (labels(68, 2)-labels(66, 2))]);
        L5Spacing = mean([(labels(65, 2)-labels(63, 2)) (labels(66, 2)-labels(64, 2))]);
        labels = cat(1, labels(1:4, :), labels);
        labels = cat(1, labels, labels(69:72, :));
        labels(1:4, 2) = max(labels(1:4, 2) - round(T1Length + T1Spacing), 1);
        labels(73:74, 2) = min(labels(73:74, 2) + round(L5Length + L5Spacing), imSize(1));
        labels(75:76, 2) = min(labels(75:76, 2) + round(L5Length*0.7 + L5Spacing), imSize(1));
        %}

        numLabels = size(labels);
        numLabels = numLabels(1);

        labels = round(labels);

        % loop through vertebrae in each image
        for k = 1:4:numLabels

            vertebra = labels(k:k+3, 1:2);

            % scaling search zone relative to vertebra size
            halfCurrentLength = round(mean([(labels(k+2, 2)-labels(k, 2)) (labels(k+3, 2)-labels(k, 2))])*length_multiplier);
            if k~=1
                CurrentSpacingTop = round(mean([(labels(k, 2)-labels(k-2, 2)) (labels(k+1, 2)-labels(k-1, 2))])*spacing_multiplier);
            else
                CurrentSpacingTop = round(mean([(labels(k+4, 2)-labels(k+2, 2)) (labels(k+5, 2)-labels(k+3, 2))])*spacing_multiplier);
            end
            if k~=(numLabels-3)
                CurrentSpacingBottom = round(mean([(labels(k+4, 2)-labels(k+2, 2)) (labels(k+5, 2)-labels(k+3, 2))])*spacing_multiplier);
            else
                CurrentSpacingBottom = round(mean([(labels(k, 2)-labels(k-2, 2)) (labels(k+1, 2)-labels(k-1, 2))])*spacing_multiplier);
            end
            smallCurrentWidth = round(mean([(labels(k+1, 1)-labels(k, 1)) (labels(k+3, 1)-labels(k+2, 1))])*width_multiplier_small);
            bigCurrentWidth = round(mean([(labels(k+1, 1)-labels(k, 1)) (labels(k+3, 1)-labels(k+2, 1))])*width_multiplier_big);

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%% top of vertebra
            line_length = vertebra(2, 1)-vertebra(1, 1)+1;

            % default line is calculated as linear connection between the
            % relevant landmark coordinates
            if vertebra(1,2) == vertebra(2,2)
                default = zeros(1, line_length) + vertebra(1,2);
            else
                default = round(vertebra(1,2):(vertebra(2,2)-vertebra(1,2))/(line_length-1):vertebra(2,2));
            end

            contour = scanVertForContour(image, default, vertebra(1, 1), vertebra(2, 1), CurrentSpacingTop, halfCurrentLength, true);
            vertebra_points = contour;

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%% bottom of vertebra
            line_length = vertebra(4, 1)-vertebra(3, 1)+1;

            if vertebra(3,2) == vertebra(4,2)
                default = zeros(1, line_length) + vertebra(3,2);
            else
                default = round(vertebra(3,2):(vertebra(4,2)-vertebra(3,2))/(line_length-1):vertebra(4,2));
            end

            contour = scanVertForContour(image, default, vertebra(3, 1), vertebra(4, 1), halfCurrentLength, CurrentSpacingBottom, false);
            vertebra_points = cat(1, vertebra_points, contour);

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%% left of vertebra
            line_length = vertebra(3, 2)-vertebra(1, 2)+1;

            if vertebra(1,1) == vertebra(3,1)
                default = zeros(1, line_length) + vertebra(1,1);
            else
                default = round(vertebra(1,1):(vertebra(3,1)-vertebra(1,1))/(line_length-1):vertebra(3,1));
            end

            contour = scanHorForContour(image, default, vertebra(1, 2), vertebra(3, 2), bigCurrentWidth, smallCurrentWidth, true);
            vertebra_points = cat(1, vertebra_points, contour);

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%% right of vertebra
            line_length = vertebra(4, 2)-vertebra(2, 2)+1;

            if vertebra(2,1) == vertebra(4,1)
                default = zeros(1, line_length) + vertebra(2,1);
            else
                default = round(vertebra(2,1):(vertebra(4,1)-vertebra(2,1))/(line_length-1):vertebra(4,1));
            end

            contour = scanHorForContour(image, default, vertebra(2, 2), vertebra(4, 2), smallCurrentWidth, bigCurrentWidth, false);
            vertebra_points = cat(1, vertebra_points, contour);

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%% construct mask using the contours
            vertebra_roi = boundary(vertebra_points, 0.85);
            BW_vertebra = poly2mask(vertebra_points(vertebra_roi,1), vertebra_points(vertebra_roi,2),imSize(1),imSize(2));

            % add vertebra roi to mask of entire spine
            BW(BW_vertebra ~= 0) = true;

        end

        imwrite(BW, roidir+filenames(m))

        % plot to test outputs
        %subplot(2,5,plotcount)
        %imshow(image)
        %hold on
        %visboundaries(BW, 'Color','g', 'LineWidth',1, 'EnhanceVisibility',false)
        %hold off
        %plotcount = plotcount + 1;

    end
end
