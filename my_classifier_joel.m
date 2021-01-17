function S = my_classifier_joel(im, parameters1, parameter2)
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


%load labels in cell format to match imds
labels = mat2cell(importdata("labels.txt"));

% image datastore size of one image: 301*225 pixels
imds = imageDatastore('imagedata');

% add labels to image datastore
imds.Labels = labels;

%size of training images
[imSizeX, imSizeY] = size(readimage(imds,1));

% build layers
layers = [
    
    % imput layer of the same size of the training images
    imageInputLayer([imSizeX,imSizeY],'name','Input layer');
    
    % create 32 convolution filters of size 3*3
    convolution2dLayer(3,32,'name','Convolution layers')

    % rectified linear activation function
    %  - output the input if it is positive, otherwise, output is zero
    reluLayer('name','ReLU')
    
    % maxpooling (size of ) mdownsamples the input to help over-fitting
    % by providing an abstracted form of the representation
    maxPooling2dLayer(2,'Stride',2,'name','Max pooling')
    
    % Fully Connected (output size = 10) layers in a neural networks are those layers
    % where all the inputs from one layer are connected to every
    % activation unit of the next layer
    fullyConnectedLayer(10, 'name','Fully connected layer')
    
    % Softmax assigns decimal probabilities to each class in a multi-class problem
    softmaxLayer('name','Softmax')
    % classification 
    classificationLayer('name','Classification layer')
    ];

    lgraph = layerGraph(layers);
    figure
    plot(lgraph)

    options = trainingOptions('sgdm','MaxEpochs',20,'InitialLearnRate',1e-4,'Verbose',false,'Plots','training-progress');
    net = trainNetwork(imds,layers,options);

end

