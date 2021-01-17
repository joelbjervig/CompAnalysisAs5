function S = my_classifier_joel(im, parameters1, parameter2)
%This classifier is not state of the art... but should give you an idea of
%the format we expect to make it easy to keep track of your scores. Input
%is the image and different parameters. Output is a 1 x 3 vector of the 
%three numbers in the image
%
%This baseline classifier tries to guess... so should score about (3^3)^-1
%on average, approx. a 4% chance of guessing the correct answer. 
%

%load data
labels = importdata("labels.txt");
S = floor(rand(1,3)*3);

% image datastore size of one image: 301*225 pixels
imds = imageDatastore('imagedata');
%size of training images
[imSizeX, imSizeY] = size(readimage(imds,1));

% build layers
layers = [
    imageInputLayer([imSizeX,imSizeY]);
    % create 32 convolution filters of size 3*3
    convolution2dLayer(3,32)
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];
end
options = 

