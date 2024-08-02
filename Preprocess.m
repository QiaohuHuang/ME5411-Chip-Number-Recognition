clc;
clear all;
close all;

%% Task 1
img = imread('charact2.bmp'); % read origin image
% figure;
% imshow(img);
% title('origin image');

% img_gray = rgb2gray(img); % turn RGB into Grey using Matlab function
img_grey = grey(img); % turn RGB into Grey using weighted average
% figure;
% imshow(img_grey);
% title('grey image')

enhence_grey = imadjust(img_grey); % contrast enhencement using matlab function

% imadjust with manually parameters
% low_in = 0.35;
% high_in = 0.65;
% low_out = 0.1;
% high_out = 0.9;
% gamma = 0.8;
% enhence_grey = imadjust(img_grey, [low_in high_in], [low_out high_out], gamma);

% histro gram equalization
% enhence_grey = histeq(img_grey);

% figure;
% imshow(enhence_grey); 
% title('enhenced image by imadjust');

%% Task 2
% filtered_grey = medfilt2(enhence_gray, [11,11]); % filter using matlab function

filtered_grey = median_filter(enhence_grey, 11); % median filter, we can try 3,5,11

% average filter
% kernel = ones(5,5)/25;
% filtered_grey = conv2(enhence_grey,kernel,'same');
% filtered_grey = uint8(filtered_grey);
% figure,imshow(filtered_grey); title("Average Filter: 5x5"); drawnow;
% 
% kernel = ones(7,7)/25;
% filtered_grey = conv2(enhence_grey,kernel,'same');
% filtered_grey = uint8(filtered_grey);
% figure,imshow(filtered_grey); title("Average Filter: 7x7"); drawnow;

figure;
imshow(filtered_grey);
title('filtered grey image by 11*11 median filter')

%% Task 3
% get region of interest - the subimage of the second line
h = imrect;
position = getPosition(h);% using mouse to get ROI
roi = imcrop(filtered_grey,position);

% XY position to crop out
% roi = get_sub(filtered_grey);

% figure;
% imshow(roi);
% title('ROI image')

%% Task 4
% threshold = 0.4;
% binary_roi = imbinarize(roi,threshold); % binary process using matlab function

binary_roi = binary(roi); 

% matlab function
% binary_roi =  imbinarize(roi);
% figure; imshow(binary_roi); title("Binary Image"); drawnow;

% Adjustable threshold
% threshold = 0.9;
% binary_roi = imbinarize(roi, threshold);
% figure; imshow(binary_roi); title("Binary Image: Adjustable threshold"); drawnow;

% figure;
% imshow(binary_roi)
% title('binary image')

se = strel('square',11); % openning to erode holes in image using matlab
opened_roi = imopen(binary_roi, se); % fufill white hole in image
% figure;
% imshow(opened_roi);
% title('binary image after opening')

% performance of my openning function is not good so I use matlab function
% opened_roi = open_operation(binary_roi, 9);

%% Task 5
edge = edge_sobel(opened_roi);
points = detectHarrisFeatures(edge); % detect possible angle point

% Thresholds and sigma for Canny edge detection
% lowThreshold = 0.01;
% highThreshold = 0.5;
% sigma = 1;
% edge = edge(opened_roi, 'Canny', [lowThreshold highThreshold], sigma);
% figure; imshow(BW); title('Character Outlines with Canny Edge Detection');

figure;
imshow(edge);
title('edge image')
hold on;
plot(points.selectStrongest(50));
%% Task 6
[row, column] = size(opened_roi);
[x,y] = getpts; % get angle point position using mouse

for i=1: length(x)
    opened_roi(round(y(i)):row,round(x(i))-3:round(x(i))+3) = 0; % black vertical line between charachter
end

figure;
imshow(opened_roi);
title('separate characters')

% [L,num] = bwlabel(binary_roi,8);% label connected component using matlab
% RGB=label2rgb(L);
% imshow(RGB);

label_img = component_label(opened_roi); % label connected component

%final_img = label2rgb(label_img); color different characters using matlab

final_img = label_to_rgb(label_img);
figure;
imshow(final_img);
title('different connceted components')

sub = separate_img(label_img);

%% Function library
function y = grey(img) % using weight average method to get grey image
    [row, column, channels] = size(img);
    img_grey = zeros(row, column);
    img_r = img(:,:,1); % red value
    img_g = img(:,:,2); % green value 
    img_b = img(:,:,3); % blue value
    img_grey = 0.2989*img_r+ 0.5870*img_g + 0.1140*img_b; % weighted average
    y = img_grey;
end

function y = median_filter(img, kernel_size) % using median filter to eliminate noise
    n = kernel_size;
    img_filter = padarray(img, [(n-1)/2 (n-1)/2]); % extent image with certain rows and colums of 0
    [row, column] = size(img_filter);
    [r,c] = size(img);
    filtered_grey = zeros(r,c);

    for i = 1: row-n+1
        for j = 1: column-n+1
            matrix = img_filter(i: i+n-1, j: j+n-1); % get matrix from img
            matrix = matrix(:); % turn matrix into a line vector
            matrix_median = median(matrix); % get median of matrix
            filtered_grey(i, j) = matrix_median; % give the median value to new image
        end
    end
    filtered_grey = uint8(filtered_grey); % trun numbers to grey values
    y =filtered_grey;
end

function y = get_sub(img)
    x = 50;
    y = 180;
    width = 900;
    height = 180;

    y = img(y:y+height, x:x+width);
end

function y = binary(img) % trun grey image into binary image
    [counts,x] = imhist(img,256); % get histrogram
    T = otsuthresh(counts); % using otsu to get threshold of binary process
    [row, column] = size(img);
    binary_roi = zeros(row, column);
    for i = 1:row % go through every pixel and compare with T
        for j = 1:column
            if img(i,j) < T*255
                binary_roi(i,j) = 0;
            else
                binary_roi(i,j) = 1;
            end
        end
    end
    y = binary_roi;
end

function y = erode(img, kernel_size) % erode to make white part smaller
    kernel = ones(kernel_size);
    [row, column] = size(img);
    opened_roi = zeros(row,column);
    img_process = padarray(img, [(kernel_size-1)/2 (kernel_size-1)/2]); % extent image with certain rows and colums of 0

    for i = 1: row
        for j = 1: column
            matrix = img_process(i: i+kernel_size-1, j: j+kernel_size-1); % get matrix from img
            out_matrix = kernel.*matrix;
            out_matrix = out_matrix(:);
            if any(out_matrix) % if there is one 0 in the out_matrix the whole matricx become 0 matrix
                opened_roi(i, j) = 0;
            else
                opened_roi(i, j) = 1;
            end
        end
    end
    
    y = opened_roi;
end

function y = dilate(img, kernel_size) % dilate to make white part bigger
    kernel = ones(kernel_size);
    [row, column] = size(img);
    opened_roi = zeros(row,column);
    img_process = padarray(img, [(kernel_size-1)/2 (kernel_size-1)/2]); % extent image with certain rows and colums of 0

    for i = 1: row
        for j = 1: column
            matrix = img_process(i: i+kernel_size-1, j: j+kernel_size-1); % get matrix from img
            out_matrix = kernel.*matrix;
            out_matrix = out_matrix(:);
            if all(out_matrix) % if there is one 1 in the out_matrix the whole matricx become 1 matrix
                opened_roi(i, j) = 0;
            else
                opened_roi(i, j) = 1;
            end
        end
    end
    
    y = opened_roi;
end

function y = open_operation(img, kernel_size)
    img1 = erode(img, kernel_size);
    img2 = dilate(img1, kernel_size);
    y = img2;
end

function y = edge_sobel(img) % using sobel kernel to detect outline
    sobelx = [-1,0,1;-2,0,2;-1,0,1]; % operator for x direction
    sobely = [-1,-2,-1;0,0,0;1,2,1]; % operator for y direction

    [row, column] = size(img);
    img_process = padarray(img, [1 1]);
    edge_img = zeros(row,column);

    for i = 1:row
        for j = 1:column
            matrix = img_process(i:i+3-1, j:j+3-1);
            x = sobelx.*matrix; 
            y = sobely.*matrix;
            Gx = sum(sum(x)); % gradient on x direction
            Gy = sum(sum(y)); % gradient on y direction
            edge_img(i,j) = sqrt(Gx^2 + Gy^2); % get the outline
        end
    end
    y = edge_img;
end

function y = component_label(img) % detect connected component
    label = 0;
    [row , column] = size(img);
    output = zeros(row+2, column+2); % extent origin image to make ture every pixel have four neighbor to go through
    for i = 1:row % go through every pixel
        for j = 1:column
            if img(i,j) == 1 % if value of pixel equal to 1, we need to check its previous neighbor 
                oi = i+1; % cause output is extended so the index need to add 1
                oj = j+1;
                % I use 8 neighbor method
                neighbors = [output(oi-1,oj-1),output(oi-1,oj),output(oi-1,oj+1),output(oi,oj-1)]; % check its previous four neighbor 
                if neighbors == [0,0,0,0] % if all 4 neighbor equal to 0, we need a new label
                    label = label + 1; % create a new label
                    output(oi,oj) = label; % make value of the pixel equal to the new label
                else % if not all neighbor equal to 0, we need to get minimum of 4 neighbor
                    label_value = []; 
                    for k = 1:4
                        if neighbors(k) ~= 0
                            label_value = [label_value, neighbors(k)];
                        end
                    end
                    output(oi,oj) = min(label_value); % find the minimum label and assign to output
                end
            end
        end
    end
    output_img = output(2:row+1,2:column+1); % crop out the output we need
    
    for i = 1:label % Then we need to go through all label to combine which are connected but have different label
        v = [];
        [r, c] = find(output_img==i); % get the positions of all pixels which have this label
        for j=1:length(r) % go through all selected pixel
            [x, y, n] = find(output_img(r(j)-1:r(j)+1, c(j)-1:c(j)+1)); % get all non 0 values of its 8 neighbor
            n = min(n); % get the minimum
            v = [v,n]; % store the minimum in v
        end
        
        v = min(v); % after go through all pixels having same value, find the minimum value of all its neighbor
        for k=1:length(r) % assign the minimum value to all pixels with this label
            output_img(r(k),c(k)) = v;
        end
    end
    y = output_img;
end

function y = label_to_rgb(img) % trun image with labels to rgb image which different component has different color
    [row, column] = size(img);
    label = [];
    % find all different labels and store them
    for i = 1: row
        for j = 1: column
            if img(i,j) ~= 0
                if ~ismember(img(i,j),label)
                    label = [label, img(i,j)]; % get all labels in the input image
                end
            end
        end
    end
    
    %generate R,G,B matrix
    y_r= zeros(row,column);
    y_g= zeros(row,column);
    y_b= zeros(row,column);

    for k = 1: length(label)
        [x,y] = find(img==label(k)); % find the position of all pixles with same label
        % generate R,G,B value
        r = randi([1 256]);
        g = randi([1 256]);
        b = randi([1 256]);
        % give R,G,B value to all pixles with same label
        for m = 1: length(x)
            y_r(x(m),y(m)) = r;
            y_g(x(m),y(m)) = g;
            y_b(x(m),y(m)) = b;
        end
    end
    
    % combine R,G,B image into RGB image
    yr = uint8(y_r);
    yg = uint8(y_g);
    yb = uint8(y_b);
    y = cat(3,yr,yg,yb);
end

function y = separate_img(img)
    [row, column] = size(img);
    label = [];
    % find all different labels and store them
    for i = 1: row
        for j = 1: column
            if img(i,j) ~= 0
                if ~ismember(img(i,j),label)
                    label = [label, img(i,j)];
                end
            end
        end
    end
    
    for l = 1:length(label)
        [r,c] = find(img==label(l)); % get the position of all pixels with this label
        max_c = max(c); % get maximum column
        min_c = min(c); % get minimum column
        
        sub_img = img(:, min_c-2:max_c+2); % crop out the input image by maximum and minimum colum
        figure
        imshow(sub_img)
        name = num2str(l);
        imwrite(sub_img, [name,'.png'])
    end
    y = 0;
end

