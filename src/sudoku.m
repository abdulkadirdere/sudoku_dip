clear all;
clc;

% read in the RGB sudoku puzzle image
sudoku_image1 = imread("../data/sudoku1.png");

% get the puzzle image in 2D
grayImage = rgb2gray(sudoku_image1);
binary_image = imbinarize(grayImage);
% imshow(binary_image);

% get complement of binary image so foreground pixels are white i.e. 1
comp_image = imcomplement(binary_image);
% imshow(comp_image);

sudoku_frame = bwareafilt(comp_image, 1, 'largest');
% imshow(sudoku_frame);

% image cropping
% find the framing square and crop the image accordingly. Label is done on
% the frame and not numbers.
labelled = bwlabel(sudoku_frame);
[row, column]=find(labelled==1);
cropped = sudoku_frame(min(row):max(row), min(column):max(column));
% imshow(cropped);

% get the location of each box in the sudoku image
stats = regionprops(imcomplement(cropped),'BoundingBox');

% store all the x & y coord of bounding boxes
bounding_box = cat(1, stats.BoundingBox);

% crop original image with numbers in it. We will work on this image
image = comp_image(min(row):max(row), min(column):max(column));


% plot the bounding boxes on the image
% imshow(~image)
% hold on
% plot(bounding_box(:,1), bounding_box(:,2),'r*')
% rectangle('Position', [bounding_box(1,1), bounding_box(1,2), bounding_box(1,3), bounding_box(1,4)], 'EdgeColor','r');
% hold off


% get number of boxes
num_box = size(bounding_box,1);
result = zeros(9,9);

% kernel for each number from 1 to 9
number_kernels = ["num1.mat","num2.mat","num3.mat","num4.mat","num5.mat","num6.mat","num7.mat","num8.mat","num9.mat"];

for i = 1:size(number_kernels,2)
    kernel = load("kernel/"+number_kernels(i)).SE;
    kernel_value = i;
    result = hitormiss(image, bounding_box, result, kernel, kernel_value, num_box);
    disp(result);
end

% write results to CSV and XLS fiels to be consumed by a sudoku solver
writematrix(result,'sudoku_puzzle.csv');
writematrix(result,'sudoku_puzzle.xls');

% display results in heatmap
figure,
heatmap(result);


%% functions

function result = hitormiss(image, bounding_box, location, kernel, kernel_value, num_box)
SE = kernel;
SE_i = uint8(~SE);

% iterate over all the boxes in the puzzle
for i=1:num_box
    % only check the boxes which hold the value zero so we skip any box
    % that has matched with a kernel
    if (location(i) == 0)
        % left, top, width, height
        roi = [bounding_box(i,1), bounding_box(i,2), bounding_box(i,3), bounding_box(i,4)];
        I2 = imcrop(image,roi);
        sub_img = uint8(I2);
%         imshow(I2);
        hitmiss = bwhitmiss(sub_img, SE, SE_i);
%         imshow(hitmiss);
        match = matching(hitmiss);
        if match == 1
            location(i) = kernel_value;
        end
    end
end
result = location;
end

% check if hitormiss found a match between kernel and value in the box
function match = matching(sub)
    [row, col] = size(sub);
    match = 0;
    for i=1:row
        for j=1:col
            if sub(i,j) == 1
                match=1;
            end
        end
    end
end
