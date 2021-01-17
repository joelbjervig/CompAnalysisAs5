function S = my_classifier(im, parameters1, parameter2)
%This classifier is not state of the art... but should give you an idea of
%the format we expect to make it easy to keep track of your scores. Input
%is the image and different parameters. Output is a 1 x 3 vector of the 
%three numbers in the image
%
%This baseline classifier tries to guess... so should score about (3^3)^-1
%on average, approx. a 4% chance of guessing the correct answer. 
%

net = googlenet;
inputSize = net.Layers(1).InputSize
% 301 225 oursize
% 224 224 their size
im=imresize(im,[224 224]);
classNames = net.Layers(end).ClassNames;
numClasses = numel(classNames);
disp(classNames(randperm(numClasses,10)))

S = floor(rand(1,3)*3);
end

