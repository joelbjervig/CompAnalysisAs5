function A = my_classifier_joel(im)
%This classifier is not state of the art... but should give you an idea of
%the format we expect to make it easy to keep track of your scores. Input
%is the image and different parameters. Output is a 1 x 3 vector of the 
%three numbers in the image
%
%This baseline classifier tries to guess... so should score about (3^3)^-1
%on average, approx. a 4% chance of guessing the correct answer. 
%

%example to use the evaluate_classifier.m file, the output should be of
%this form: [2,1,3] i.e an array of length 3 containg the classified digits
%[net,accuracy,info] = trainingNet();
load net;
C = char((classify(net,im)));
A = [str2num(C(1)) str2num(C(2)) str2num(C(3))];

