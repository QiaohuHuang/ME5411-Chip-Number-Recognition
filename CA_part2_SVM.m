clc;
clear all;
close all;

% load dataset
path = '.\dataset\';
imds = imageDatastore(path,...
       "IncludeSubfolders",true,...
       "FileExtensions",'.png',...
       'LabelSource','foldernames');

numTrainingFiles = round(254*0.75); % set training set and testing set
[trainSet,testSet] = splitEachLabel(imds,numTrainingFiles,'randomize');

t = templateSVM('SaveSupportVectors',true, 'Standardize', true, 'KernelFunction','polynomial', ...
    'KernelScale', 'auto','Verbose', 1);  % set parameters of SVM

trainFeatures = zeros(length(trainSet.Labels), 128*128); % create features of each input image
for i = 1:length(trainSet.Labels)
    img = readimage(trainSet, i);
    trainFeatures(i, :) = img(:)'; % turn the image into a line and use pixel values as features
end

tic;
svm_model = fitcecoc(trainFeatures, trainSet.Labels,'Learner', t); % train SVM model
t1 = toc;
save('svm.mat','svm_model');

testFeatures = zeros(length(testSet.Labels), 128*128); % test trained SVM model
for i = 1:length(testSet.Labels) 
    img = readimage(testSet, i);
    testFeatures(i, :) = img(:)';
end

% accuracy on testing set
predicted_labels = predict(svm_model, testFeatures);

accuracy = sum(predicted_labels == testSet.Labels) / numel(testSet.Labels);
fprintf('Test_Accuracy: %.2f%%\n', accuracy * 100);

% % accuracy on training set
% predicted_labels = predict(svm_model, trainFeatures);
% 
% accuracy = sum(predicted_labels == trainSet.Labels) / numel(trainSet.Labels);
% fprintf('Train_Accuracy: %.2f%%\n', accuracy * 100);

