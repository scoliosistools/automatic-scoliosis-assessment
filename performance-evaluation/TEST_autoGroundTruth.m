clear
close all

% directory of x-rays
imdir = "C:\data\ScoliosisProject\BoostNet_datasets\boostnet_labeldata\data\test";

% directory of spineMasks
maskdir = 'C:\data\ScoliosisProject\BoostNet_datasets\Predictions\SpineMasks - ground-truth endplates\';
files = dir(fullfile(maskdir, '*.jpg'));

% read spreadsheet of filenames
filenames = readmatrix('C:\data\ScoliosisProject\BoostNet_datasets\boostnet_labeldata\labels\test\filenames.csv', 'ExpectedNumVariables', 1, 'OutputType', 'string', 'Delimiter',',');


numImages = size(filenames);
numImages = numImages(1);

indicesToProcess = 1:numImages;

plotting = true; % toggle this for plotting vs saving results
if plotting
    plotcount = 1;
    randomSample = randi(128, 1, 5); %[3 11 13 48];%[16 35 48 85 108 12 36 62 108];
    indicesToProcess = randomSample;
    figure(1)
    sgtitle('Random Sample of Images & Corresponding Ground-truth Vertebral Segmentations')
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
    spineMask = imbinarize(spineMask(:,:,1));
    % Plotting
    if plotting
        figure(1)
        subplot(2,5,plotcount)
        imshow(xray)
        subplot(2,5,plotcount+5)
        hold on
        imshow(xray)
        visboundaries(spineMask, 'Color','b', 'LineWidth',0.5, 'EnhanceVisibility',false)
        hold off
        plotcount = plotcount + 1;
    end
end