clc;
clear all;
close all;

% load dataset
path = '.\dataset\';
imds = imageDatastore(path,...
       "IncludeSubfolders",true,...
       "FileExtensions",'.png',...
       'LabelSource','foldernames');

% randomly show some images from train data
% figure
% numImages = 254*7;
% perm = randperm(numImages,20);
% for i = 1:20
%     subplot(4,5,i);
%     imshow(imds.Files{perm(i)});
%     drawnow;
% end

% divide train/test set by 4/1
numTrainingFiles = round(254*0.75);
[imdsTrain,imdsTest] = splitEachLabel(imds,numTrainingFiles,'randomize');

% define network architecture
layers = [ ...
    imageInputLayer([128 128 1]) % the size of input image

    convolution2dLayer(3,32,'Padding','same') % choose the size of filters and the numbers of them
    reluLayer % activation layer
    maxPooling2dLayer(2,'Stride',2) % max pooling layer, why greater stride causes better perform????

    fullyConnectedLayer(7) % fully connected layer to learn the features
    softmaxLayer % activation layer
    classificationLayer]; % caculate cross entrophy loss

% set hyperparameters
options = trainingOptions('sgdm', ...
    'MaxEpochs',20,... % set maximum learning epochs
    'InitialLearnRate',1e-4, ... % set initial learning rate
    'Verbose',false, ...
    'Plots','training-progress'); % visualize the training process

% train network
net = trainNetwork(imdsTrain,layers,options);

% test network
YPred = classify(net,imdsTest);
YTest = imdsTest.Labels;

accuracy = sum(YPred == YTest)/numel(YTest);

%% feature map visualization
% %analyzeNetwork(net);
% layer = 2;
% name = net.Layers(layer).Name;
% channels = 1:20;
% I = deepDreamImage(net,name,channels, ...
%     'PyramidLevels',1);
% 
% figure
% I = imtile(I,'ThumbnailSize',[64 64]);
% imshow(I)
% title(['Layer ',name,' Features'],'Interpreter','none')

%% save the trained network
save('one_layer_cnn.mat','net')





