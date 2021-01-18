function A = my_classifier_joel(im,net)
%[net, info] = trainingNet();
%load net;
C = char((classify(net,im)));
A = [str2num(C(1)) str2num(C(2)) str2num(C(3))];
end

