clear;clc;close all
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
    imwrite(cat(3,I2,I2,I2),names{i})
%     figure;imshow(I2)
%     J2 = imdilate(I2,SE);
%     I3=abs(double(I2)-double(J2));
%     I3=uint8((1-I3/255)*255);
%     BW = imbinarize(I3);
    
%     BW = imbinarize(I);
%     BW_comp = imcomplement(BW);
%     BW2_comp = bwareaopen(BW_comp,30,8);
%     BW2 =imcomplement(BW2_comp);
% %     figure;imshowpair(BW,BW2,'montage')
%     imwrite(uint8(BW2)*255,names{i})
end
