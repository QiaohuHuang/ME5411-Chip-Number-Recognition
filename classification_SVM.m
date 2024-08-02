clc;
clear all;
close all;

load('svm.mat');

example = imread('img001-00766.png');
[r,c] = find(~example);
rmax = max(r);
rmin = min(r);
cmax = max(c);
cmin = min(c);

rsize = rmax-rmin; % get the row size of character in train set
csize = cmax-cmin; % get the column size of character in train set

r_padding = (128-rsize)/2;
c_padding = (128-csize)/2;

true = ['HD44780A00']
pred = [];
for i = 1:10
    image = imread([num2str(i,'%d'),'.png']);
    % image pre-processing
    image = imresize(image,[rsize csize]); % resizing
    image = padarray(image,[r_padding c_padding]); % padding
    image = imbinarize(image);
    image = ~image*255; % turn black into white and white into black and map into 0 and 255
    features = image(:)';
    predi = predict(svm_model, features);
    pred = [pred predi];
end
pred
