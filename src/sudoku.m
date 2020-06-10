clear all;
clc;

% Read in the RGB Sudoku Puzzle Image
sudoku_image = imread("../data/sudoku1.png");

% Preprocessing of input image to prepare for image processing techniques
[image, bounding_box] = preImageProcessing(sudoku_image);

% Plot marked corners of boxes within the frame, verifying bounding boxes
plotImageBoundingBox(image, bounding_box);

% Apply our optical character recognition function, buit using Image
% Processing tools to find all characters within the puzzel
result = OCR(image, bounding_box);

writeResults(result);
% display results in heatmap and save the heatmap as an image
figure,
heatmap(result);
saveas(gcf,'output/sudoku_puzzle.png')


%% functions

function [image, bounding_box, comp_image] = preImageProcessing(sudoku_image)
    % Convert to grey scale image
    grayImage = rgb2gray(sudoku_image);
    % Convert to a binary image
    binary_image = imbinarize(grayImage);

    % Take complement of image, format suitable for BW. Foreground pixels
    % must be white
    comp_image = imcomplement(binary_image);

    % Extract the connected components, specifically the largest connected
    % component. This is the frame and the interconnected horizontal and
    % vertical lines
    sudoku_frame = bwareafilt(comp_image, 1, 'largest');

    % Crop the frame, remove any white space outside of the frame of the
    % puzzle
    [row, column]=find(sudoku_frame==1);
    cropped = sudoku_frame(min(row):max(row), min(column):max(column));
    % imshow(~cropped);

    % Perform Connected Component search again on the frame this time. This
    % will give us the number of boxes within the frame.
    % Region props allows us to detect the co-ordinates of each connected 
    % component. i.e. Location of each box in the sudoku frame
    stats = regionprops(imcomplement(cropped),'BoundingBox');

    % Store all the x & y coord of bounding boxes, store in a list
    bounding_box = cat(1, stats.BoundingBox);

    % Crop binarised image with numbers in it. We will work on this going
    % forward image.
    image = comp_image(min(row):max(row), min(column):max(column));
end

function plotImageBoundingBox(image, bounding_box)
    % Plot the bounding boxes on the image, so that we can see points of
    % the corners of the individual boxes
    imshow(~image)
    hold on
    plot(bounding_box(:,1), bounding_box(:,2),'r*')
    rectangle('Position', [bounding_box(1,1), bounding_box(1,2), bounding_box(1,3), bounding_box(1,4)], 'EdgeColor','r');
    hold off
end

function result = hitormiss(image, bounding_box, box_value, kernel, kernel_value, num_box)
% Current SE
SE = kernel;
SE_i = uint8(~SE);

% Iterate over all the boxes in the puzzle
for i=1:num_box
    % Only check the boxes which hold the value zero.
    % Optimisation technique so we skip any box that has already 
    % matched with a kernel
    if (box_value(i) == 0)
        % Specify the current box or ROI
        roi = [bounding_box(i,1), bounding_box(i,2), bounding_box(i,3), bounding_box(i,4)];
        % Extract it from the original image
        I2 = imcrop(image,roi);
        % Convert to int
        sub_img = uint8(I2);
        % Apply hit or miss to extracted int image
        hitmiss = bwhitmiss(sub_img, SE, SE_i);
        % Check for match
        match = check_match(hitmiss);
        % If true
        if match == 1
            box_value(i) = kernel_value;
        end
    end
end
result = box_value;
end

% check if hitormiss found a match between kernel and value in the box
function match = check_match(sub)
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

function result = OCR(image, bounding_box)
    % Get number of boxes
    num_box = size(bounding_box,1);
    % Create the resulting matrix. Tracking array that we will display in
    % the end
    result = zeros(9,9);
    
    % Import a predifined Kernel for each number from 1 to 9
    % We have created these Kernels manually
    kernels = ["num1.mat","num2.mat","num3.mat","num4.mat","num5.mat","num6.mat","num7.mat","num8.mat","num9.mat"];
    
    % Apply a hit or miss to each of the blocks using every Kernel
    for i = 1:size(kernels,2)
        kernel = load("kernel/"+kernels(i)).SE;
        kernel_value = i;
        result = hitormiss(image, bounding_box, result, kernel, kernel_value, num_box);
        % Display the Sudoku Puzzle
        disp(result);
    end
end

function writeResults(result)
    % write results to CSV and XLS fiels to be consumed by a sudoku solver
    writematrix(result,'output/sudoku_puzzle.csv');
    writematrix(result,'output/sudoku_puzzle.xls');
end