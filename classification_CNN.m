clc;
clear all;
close all;

load('three_layer_cnn.mat'); % load the pretrained three layer network
%load('four_layer_cnn.mat'); % load the pretrained three layer network

%% imresize and padding method
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
    y = classify(net, image);
    pred = [pred, y];
end
pred

%% classify all images with directly resize to 128*128
% true = ['HD44780A00']
% pred = [];
% for i = 1:10
%     image = imread([num2str(i,'%d'),'.png']);
%     % image pre-processing
%     image = imresize(image,[128 128]); % resizing
%     image = imbinarize(image);
%     image = ~image*255; % turn black into white and white into black and map into 0 and 255
%     y = classify(net, image);
%     pred = [pred, y];
% end
% pred

%% classify one image
% image = imread('1.png');
% image = imresize(image,[100 80]); % resizing
% image = padarray(image,[14 24]); % padding
% image = imbinarize(image);
% image = ~image*255; 
% figure
% imshow(image)
% y = classify(net, image)

