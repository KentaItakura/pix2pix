function img2sketch()
    info=dir('CelebA-HQ-img/*.jpg');
    names={info.name};
    imDir=fullfile(pwd,"CelebA-HQ-img");
    mkdir('CelebA_Line')
    SE = strel('square',3);
    cd('CelebA_Line')
    for i=1:numel(names)
        I=imread(fullfile(imDir,names{i}));
        I=rgb2gray(I);
        J = imdilate(I,SE);
        I2=abs(double(I)-double(J));
        I2=uint8((1-I2/255)*255);
        imwrite(I2,names{i})
    end
    cd ../
end
