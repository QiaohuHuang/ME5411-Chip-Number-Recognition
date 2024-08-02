clc;
clear all;
close all;

load('b1.mat');
load('b2.mat');
load('b3.mat');

load('w1.mat');
load('w2.mat');
load('w3.mat');

load('train_data.mat')
load('train_label.mat')
load('test_label'); 
load('test_data'); 

%% test on test dataset
% true = 0;
% for i = 1:1337
%     XTest = train_data(:,i);
%     XTest = double(XTest);
%     YTest = train_label(:,i);
% 
%     Z1_test = W1 * XTest + b1;
%     A1_test = relu(Z1_test);
%     Z2_test = W2 * A1_test+b2;
%     A2_test = relu(Z2_test);
%     Z3_test = W3 * A2_test + b3;
%     A3_test = softmax(Z3_test);
% 
%     YPred = A3_test;
%     [vp,pp] = max(YPred);
%     [v, p] = max(YTest);
%     if pp == p
%         true = true+1;
%     end
% end
% 
% accuracy = true/1337;


%% classify images from Image1
true_value = ['0','4','7','8','A','D','H'];

image = imread('2.png');
figure;
imshow(image);
image = imresize(image,[100 80]);
image = padarray(image,[14 24]);
image = imbinarize(image);
image = ~image;

input = image(:);

Z1 = W1 * input + b1;
A1 = relu(Z1);
Z2 = W2 *A1 + b2;
A2 = relu(Z2);
Z3 = W3 * A2 + b3;
A3 = softmax(Z3);

[value, position] = max(A3);
output = true_value(position);
title(output);

function y = relu(x) % relu
    y = max(0, x);
end