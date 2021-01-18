for k = 1:length(labels)
    % read from original folder and preform median filter and binarize
    im = imbinarize(medfilt2(imread(sprintf('imagedata/train_%04d.png', k)),[6,6]), 0.6);
    % write to new folder
    imwrite(im,sprintf('imagedata_processed/train_%04d.png',k))
end