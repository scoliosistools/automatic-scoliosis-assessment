% script to test all functionality of the designed system using the boostnet data

% toggle this for plotting a random sample vs saving results for all test data
plotting = false;

clear
close all

addpath('..\clinical-assessment-algorithms');

% directory of x-rays
imdir = "..\data\boostnet_labeldata\data\test";

% directory of spineMasks
maskdir = '..\data\PredictionsVsGroundTruth\SpineMasks\';
files = dir(fullfile(maskdir, '*.jpg'));

% read spreadsheet of filenames
filenames = readmatrix('..\data\boostnet_labeldata\labels\test\filenames.csv', 'ExpectedNumVariables', 1, 'OutputType', 'string', 'Delimiter',',');

processedMaskDestinationDir = '..\data\PredictionsVsGroundTruth\SpineMasks_Processed\';
endplatesDestinationDir = '..\data\PredictionsVsGroundTruth\Endplates\';

gtLandmarks = load('..\data\FixedSpineWebData\fixedTestingLandmarks.mat');
gtLandmarks = gtLandmarks.landmarks;

numImages = size(filenames);
numImages = numImages(1);

allEndplateSlopes = zeros(numImages,34);
gtAllEndplateSlopes = zeros(numImages,34);

allAngles = zeros(numImages,3);
gtAllAngles = zeros(numImages,3);

allLenkeCurveTypes = zeros(numImages,1);
gtAllLenkeCurveTypes = zeros(numImages,1);

allLenkeCurveTypeProbabilities = zeros(numImages,6);
gtAllLenkeCurveTypeProbabilities = zeros(numImages,6);

indicesToProcess = 1:numImages;

if plotting
    plotcount = 1;
    randomSample = randi(128, 1, 5);
    indicesToProcess = randomSample;
    figure(1)
    sgtitle('Random Sample of Images & Corresponding Vertebral Segmentations')
    figure(2)
    sgtitle('Random Sample of Vertebral Segmentations & Corresponding Fitted Endplates')
    figure(3)
    sgtitle('Random Sample of Fitted Endplates & Corresponding Cobb Angles')
    figure(4)
    sgtitle('Random Sample of Cobb Angles & Corresponding Lenke Curve Type Probabilities')
    figure(5)
    sgtitle('Random Sample of End-to-End Performance')
end

for n = indicesToProcess
    
    % read through files in correct order by checking filename
    for filecounter =  1:length(files)
        if files(filecounter).name == filenames(n)
            xray = imread(fullfile(imdir, files(filecounter).name));
            spineMask = imread(fullfile(maskdir, files(filecounter).name));
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%% process spine mask
    spineMask = imresize(spineMask, size(xray));
    % Plotting
    if plotting
        figure(5)
        subplot(5,5,plotcount)
        imshow(xray)
    end
    
    spineMask = processSpineMask(spineMask);
    
    % Saving
    if ~plotting
        imwrite(spineMask, processedMaskDestinationDir+filenames(n));
    end
    
    % Plotting
    if plotting
        figure(5)
        subplot(5,5,5+plotcount)
        hold on
        imshow(xray)
        visboundaries(spineMask, 'Color','b', 'LineWidth',0.5, 'EnhanceVisibility',false)
        hold off
        
        figure(1)
        subplot(2,5,plotcount)
        imshow(xray)
        subplot(2,5,plotcount+5)
        hold on
        imshow(xray)
        visboundaries(spineMask, 'Color','b', 'LineWidth',0.5, 'EnhanceVisibility',false)
        hold off
        
    end
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%% fit endplates
    [endplateLandmarks, centroidsTop2Bottom] = fitEndplates(spineMask);
    
    gtLandmarksScaled = squeeze(gtLandmarks(n,:,:)).*fliplr(size(xray));
    % Saving
    if ~plotting
        endplateSlopes = zeros(1,(length(endplateLandmarks)/2));
        for k = 1:(length(endplateLandmarks)/2)
            landmarkPos = (k-1)*2+1;
            endplateSlopes(1,k) = (endplateLandmarks(landmarkPos+1,2) - endplateLandmarks(landmarkPos,2)) / (endplateLandmarks(landmarkPos+1,1) - endplateLandmarks(landmarkPos,1));
        end
        
        gtEndplateSlopes = zeros(1,(length(gtLandmarksScaled)/2));
        for k = 1:(length(gtLandmarksScaled)/2)
            landmarkPos = (k-1)*2+1;
            gtEndplateSlopes(1,k) = (gtLandmarksScaled(landmarkPos+1,2) - gtLandmarksScaled(landmarkPos,2)) / (gtLandmarksScaled(landmarkPos+1,1) - gtLandmarksScaled(landmarkPos,1));
        end
        
        alignedEndplateSlopes = alignSlopeVectors(gtEndplateSlopes, endplateSlopes);
        
        allEndplateSlopes(n,:) = rad2deg(atan(alignedEndplateSlopes));
        gtAllEndplateSlopes(n,:) = rad2deg(atan(gtEndplateSlopes));
        
        %save(endplatesDestinationDir+filenames(n)+'.mat', 'endplateLandmarks', 'centroidsTop2Bottom');
    end
    
    % Plotting
    if plotting
        figure(5)
        subplot(5,5,10+plotcount)
        imshow(xray)
        hold on
        for k = 1:2:length(endplateLandmarks)-1
            plot(endplateLandmarks([k k+1],1), endplateLandmarks([k k+1],2), 'g');
        end
        hold off
        
        figure(2)
        subplot(2,5,plotcount)
        hold on
        imshow(xray)
        visboundaries(spineMask, 'Color','b', 'LineWidth',0.5, 'EnhanceVisibility',false)
        hold off
        subplot(2,5,plotcount+5)
        hold on
        imshow(xray)
        for k = 1:2:length(endplateLandmarks)-1
            plot(endplateLandmarks([k k+1],1), endplateLandmarks([k k+1],2), 'g');
        end
        hold off
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%% calculate Cobb angles
    nullAngleLocations = zeros(4,1); % used to allow for manual setting of angle locations
    [cobbAngles, apicalVertebrae, angleLocations, cobbEndplates] = calculateCobbAngles(endplateLandmarks, centroidsTop2Bottom, nullAngleLocations);
    allAngles(n,:) = cobbAngles';
    
    [gtCobbAngles, gtApicalVertebrae, gtAngleLocations, gtCobbEndplates] = calculateCobbAngles(gtLandmarksScaled, centroidsTop2Bottom, nullAngleLocations);
    gtAllAngles(n,:) = gtCobbAngles';

    
    % Plotting
    if plotting
        figure(5)
        subplot(5,5,15+plotcount)
        imshow(xray)
        hold on
        for k = 1:3
            if k == 1
                style1 = 'c';
            elseif k ==2
                style1 = 'r';
            else
                style1 = 'y';
            end
            plot(cobbEndplates(1:2,1,k), cobbEndplates(1:2,2,k), style1);
            plot(cobbEndplates(3:4,1,k), cobbEndplates(3:4,2,k), style1);
        end
        hold off
        
        figure(3)
        subplot(2,5,plotcount)
        hold on
        imshow(xray)
        for k = 1:2:length(endplateLandmarks)-1
            plot(endplateLandmarks([k k+1],1), endplateLandmarks([k k+1],2), 'g');
        end
        hold off
        subplot(2,5,plotcount+5)
        hold on
        imshow(xray)
        for k = 1:3
            if k == 1
                style1 = 'c';
                %style2 = 'c:';
            elseif k ==2
                style1 = 'r';
                %style2 = 'r:';
            else
                style1 = 'y';
                %style2 = 'y:';
            end
            plot(cobbEndplates(1:2,1,k), cobbEndplates(1:2,2,k), style1);
            plot(cobbEndplates(3:4,1,k), cobbEndplates(3:4,2,k), style1);
            %plot(gtCobbEndplates(1:2,1,k), gtCobbEndplates(1:2,2,k), style2);
            %plot(gtCobbEndplates(3:4,1,k), gtCobbEndplates(3:4,2,k), style2);
        end
        hold off
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%% classify Lenke curve type
    [curveType, curveTypeProbabilities] = classifyLenkeCurveType(cobbAngles');
    [gtCurveType, gtCurveTypeProbabilities] = classifyLenkeCurveType(gtCobbAngles');
    
    allLenkeCurveTypes(n,1) = curveType;
    gtAllLenkeCurveTypes(n,1) = gtCurveType;
    
    allLenkeCurveTypeProbabilities(n,:) = curveTypeProbabilities;
    gtAllLenkeCurveTypeProbabilities(n,:) = gtCurveTypeProbabilities;
    
    % Plotting
    if plotting
        figure(5)
        subplot(5,5,20+plotcount)
        barh(curveTypeProbabilities)
        xlim([0 1])
        xlabel('Probability')
        ylabel('Curve Type')
        
        figure(4)
        subplot(2,5,plotcount)
        imshow(xray)
        hold on
        for k = 1:3
            if k == 1
                style1 = 'c';
            elseif k ==2
                style1 = 'r';
            else
                style1 = 'y';
            end
            plot(cobbEndplates(1:2,1,k), cobbEndplates(1:2,2,k), style1);
            plot(cobbEndplates(3:4,1,k), cobbEndplates(3:4,2,k), style1);
        end
        hold off
        
        subplot(2,5,plotcount+5)
        barh(curveTypeProbabilities)
        xlim([0 1])
        xlabel('Probability')
        ylabel('Curve Type')
        

        plotcount = plotcount + 1;
    end
    
end

% write to csv file
if ~plotting
    csvwrite('..\data\PredictionsVsGroundTruth\EndplateSlopes.csv',allEndplateSlopes);
    csvwrite('..\data\PredictionsVsGroundTruth\EndplateSlopes_GroundTruthEndplates.csv',gtAllEndplateSlopes);
    
    csvwrite('..\data\PredictionsVsGroundTruth\Angles.csv',allAngles);
    csvwrite('..\data\PredictionsVsGroundTruth\Angles_GroundTruthEndplates.csv',gtAllAngles);
    
    csvwrite('..\data\PredictionsVsGroundTruth\LenkeCurveTypes.csv',allLenkeCurveTypes);
    csvwrite('..\data\PredictionsVsGroundTruth\LenkeCurveTypes_GroundTruthEndplates.csv',gtAllLenkeCurveTypes);
    
    csvwrite('..\data\PredictionsVsGroundTruth\LenkeCurveTypeProbabilities.csv',allLenkeCurveTypeProbabilities);
    csvwrite('..\data\PredictionsVsGroundTruth\LenkeCurveTypeProbabilities_GroundTruthEndplates.csv',gtAllLenkeCurveTypeProbabilities);
end


