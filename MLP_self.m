clc;
clear all;
close all;

tic;
%% get data
load('train_data'); % each column is a image vector
load('test_data'); 
load('train_label'); % each column is a label
load('test_label'); 

%% network 
% parameters
inputSize = 128*128;
hiddenSize1 = 3000;
hiddenSize2 = 300;
outputSize = 7;
learningRate = 0.001;
numEpochs = 200;

% initial weights and bias
W1 = randn(hiddenSize1, inputSize)*0.05;
b1 = zeros(hiddenSize1, 1);
W2 = randn(hiddenSize2, hiddenSize1)*0.05;
b2 = zeros(hiddenSize2, 1);
W3 = randn(outputSize, hiddenSize2)*0.05;
b3 = zeros(outputSize, 1);

% train
Loss_set = [];
for epoch = 1:numEpochs
    
    lr = learningRate*(0.99^epoch);
    idx = randi([1 1337]);
    Xtrain = train_data(:,idx);
    Xtrain = double(Xtrain);

    [value,position] = max(Xtrain);

    Y = train_label(:,idx);
    Y = double(Y);
    
    % forward
    Z1 = W1 * Xtrain + b1;
    A1 = relu(Z1);
    Z2 = W2 * A1 + b2;
    A2 = relu(Z2);
    Z3 = W3 * A2 + b3;
    A3 = softmax(Z3);
    
    % loss
    loss = -sum(sum(Y .* log(A3))) / size(Y, 2);
    
    % backward
    delta3 = A3-Y;
    dW3 = (1 / size(Y, 2)) * delta3 * A2';
    db3 = (1 / size(Y, 2)) * sum(delta3, 2);
    delta2 = (W3' * delta3) .* reluGradient(Z2);
    dW2 = (1 / size(Y, 2)) * delta2 * A1';
    db2 = (1 / size(Y, 2)) * sum(delta2, 2);
    delta1 = (W2' * delta2) .* reluGradient(Z1);
    dW1 = (1 / size(Y, 2)) * delta1 * Xtrain';
    db1 = (1 / size(Y, 2)) * sum(delta1, 2);
    
    % update weights and bias
    W1 = W1 - lr * dW1;
    b1 = b1 - lr * db1;
    W2 = W2 - lr * dW2;
    b2 = b2 - lr * db2;
    W3 = W3 - lr * dW3;
    b3 = b3 - lr * db3;
    
    % print loss
    fprintf('Epoch %d, Loss: %.4f\n', epoch, loss);
    Loss_set = [Loss_set,loss];

end

figure;
x = 1:1:numEpochs;
plot(x,Loss_set);

t = toc;
%% test on test dataset
% true = 0;
% for i = 1:441
%     XTest = test_data(:,i);
%     XTest = double(XTest);
%     YTest = test_label(:,i);
%     Z1_test = W1 * XTest + b1;
%     A1_test = relu(Z1_test);
%     Z2_test = W2 * A1_test+b2;
%     A2_test = relu(Z2_test);
%     Z3_test = W3 * A2_test + b3;
%     A3_test = softmax(Z3_test);
%     YPred = A3_test;
%     [vp,pp] = max(YPred);
%     [v, p] = max(YTest);
%     if pp == p
%         true = true+1;
%     end
% end
% 
% accuracy = true/441


%% function library
function y = relu(x) % relu
    y = max(0, x);
end

function y = reluGradient(x) % gradient of relu
    y = double(x > 0);
end

function y = softmax(x) % softmax
    exp_x = exp(x - max(x));
    y = exp_x ./ sum(exp_x);
end





