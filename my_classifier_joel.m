function A = my_classifier_joel(im,net)
%[net, info] = trainingNet();
%load net;
I = medfilt2(im, [7 7]);
I = imbinarize(I, 0.6);
C = char((classify(net,I)));
A = [str2num(C(1)) str2num(C(2)) str2num(C(3))];
end

