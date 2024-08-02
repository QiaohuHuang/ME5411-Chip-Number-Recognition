clc;
clear all;
close all;

% load dataset
path = '.\dataset\';
imds = imageDatastore(path,...
       "IncludeSubfolders",true,...
       "FileExtensions",'.png',...
       'LabelSource','foldernames');

% figure
% numImages = 254*7;
% perm = randperm(numImages,20);
% for i = 1:20
%     subplot(4,5,i);
%     imshow(imds.Files{perm(i)});
%     drawnow;
% end

% divide train/test set
numTrainingFiles = round(254*0.75);
[imdsTrain,imdsTest] = splitEachLabel(imds,numTrainingFiles,'randomize');

% define network architecture
layers = [ ...
    imageInputLayer([128 128 1])
    fullyConnectedLayer(3000)
    reluLayer
    fullyConnectedLayer(300)
    reluLayer
    fullyConnectedLayer(7)
    softmaxLayer
    classificationLayer];

% set hyperparameters
options = trainingOptions('sgdm', ...
    'MaxEpochs', 5, ...
    'MiniBatchSize', 2, ... % affecting the performence a lot
    'InitialLearnRate', 0.0001, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% train network
net = trainNetwork(imdsTrain,layers,options);

% test network
YPred = classify(net,imdsTest);
YTest = imdsTest.Labels;

accuracy = sum(YPred == YTest)/numel(YTest);
