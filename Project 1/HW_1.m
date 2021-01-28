%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Project 1: Read and display an image(raw file) and Affine transformation 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Q1.Determine a process where you can acquire an image and use Matlab commands to get that image into Matlab. Make sure you can read in gray-scale images. 

% In the first part, we try to read an image in matlab and check if the
% image can be viewed successfully. Three different image file formats were
% used:
%1).3FR
%2).cr2
%3).nef 
%4).dng
% A sample image of the format raw file can be viewed by using the following code:
%% Reading a raw image file
row=576;  col=768;
fin=fopen('Human.raw');
I=fread(fin, col*row*3,'uint8=>uint8'); %// Read in as a single byte stream
I = reshape(I, [col row 3]); %// Reshape so that it's a 3D matrix - Note that this is column major
Ifinal = flipdim(imrotate(I, -90),2); % // The clever transpose
imshow(Ifinal);
fclose(fin);
%% Using imread for cr2 format:-
% Now we try to view the image using imread. On running the code,the image can be viewed successfully.
z = imread('Bike.cr2');
info = imfinfo('Bike.cr2')
figure
image(z)
%% Using fread function:
info = imfinfo('converted_images_1.dng')
fid=fopen('converted_images_1.dng');
A=fread(fid,[8896,5920],'uint16');
fclose(fid); 
A=A';
imagesc(A);

% On running the code we see that the image had a lot of noise and hence
% the image was not clear.
%% Convert to grayscale":
I = rgb2gray(z);
figure
imshow(I)
%% Using imread for nef format:
% Now, we try to open an image of the file format .NEF. On running the
% code, the image can be viewed.
warning('off')
o = imread('baby.nef');
info_1 = imfinfo('baby.nef')
figure
image(o)
%% Using fread function:
fid=fopen('baby.nef'); 
B=fread(fid,[320,212],'uint16');
fclose(fid); 
B=B';
colormap(jet);
imagesc(B);
% On running the code we see that the image had some noise even though it was almost visible.
%% Convert to grayscale:
J = rgb2gray(o);
figure
imshow(J)
%% Using imread for DNG format:
% Now, we try to open an image of the file format .DNG. On running the
% code, the image can be viewed.
d = imread('Tree.DNG');
figure
info_2 = imfinfo('Tree.DNG')
image(d)
%% Using fread function:

fid=fopen('Tree.DNG'); 
C=fread(fid,[5216,3472],'uint16');
fclose(fid); 
C=C';
colormap(jet);
imagesc(C);

% On running the code we see that the image is not very clear on using the
% fread function
%% Convert to grayscale:
K = rgb2gray(d);
figure
imshow(K)
%% Using imread for .3FR format:

dd = imread('Raw2.3FR');
figure
info_3 = imfinfo('Raw2.3FR')
image(dd)
%% Using fread function:
fid=fopen('Raw2.3FR'); 
CC=fread(fid,[8384,6304],'uint16');
fclose(fid); 
MM=(CC)';
imagesc(MM);
colormap(jet);
% On running the code we see that the image is not very clear on using the
% fread function
%% Convert to grayscale;

KK = rgb2gray(dd);
figure
imshow(KK)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Q2.Read in an image and display it.

clims=[0,10000];
figure
imagesc(A, clims)
%% To grayscale using colormap
colormap(jet)
%colormap(bone)
% Colormap function did not work as this is a 3D image.
%% To grayscale 
II = rgb2gray(Ifinal);
figure
imshow(II)
%% Q3.Read in an image and scale the intensity values from 0 – 255.

%Read an image:
fid=fopen('Tree.DNG'); 
C=fread(fid,[5216,3472],'uint16');
fclose(fid); 
C=C';
imagesc(C);

%Scale the intensity values to 0-255:
C = C.*3;   % some intensity values could be > 255
maxPix = max(max(C)); % find the maximum
minPix = min(min(C)); % find the minimum

% make it to the range between 0 to 1 and then multiply by 255.
C = ((C + minPix)/maxPix)*255;
image(C);

% Add noise to the image
% Take the average of N noisy versions of the original (using independent noise samples!)

N = [10,20,30,40,50];
SumOfImages = zeros(size(C));
mean = 20;
variance = 20;
l = length(N);
MSE=zeros(l,1);

for i=1:l

    for x=1:N(i)
        SumOfImages= SumOfImages + imnoise(C,'gaussian',mean,variance);
    end
    Avg = SumOfImages./i;
    MSE(i) = immse(C,Avg);
end
fprintf('\n The mean-squared error is %0.4f\n', MSE);
plot(N,MSE)
title('N vs MSE');
xlabel('N');
ylabel('MSE');
%% Q4.Affine transformation. Read in an image, and rotate and scale it by 35 degrees and a scale of 0.7 in all directions.

%Read an Image
C=imread('converted_images_1.dng');
% Rotate and Scale the image
theta = 35;
tform= affine2d([cosd(theta) -sind(theta) 0; sind(theta) cosd(theta) 0; 0 0 1]);
J = imwarp(C,tform);
colormap(gray)
imagesc(J);
%%
% Scaling Image
scale = affine2d([0.7 0 0;0 0.7 0;0 0 1]);
L = imwarp(C,scale);
imagesc(L);
%% Q/A

%% 1. A brief write-up of what you did.

%A)
% In the first part, we try to read an image in matlab and check if the
% image can be viewed successfully. Three different image file formats were
% used:
%1).3FR
%2).cr2
%3).nef 
%4).dng

% In the second part of the question, we try to use the colormap function
% to convert the image to grayscale.Although, when colourmap(grayscale)
% function was used, at some instances, the images were not being plotted.
% Hence I used the rgb2gray function and plotted the grayscale version of
% the image.

% In the third question, a value of 20 was provided for both the mean and
% variance and the image was scaled between 0-255 by using the imnoise
% function. Then MSE was computed using the immse function.

% In the fourth question, an image was read and rotated to an angle of 30
% degrees. Another image was scaled to 0.7 in all directions.

%% 2.An image displayed with 32 and 255 intensity levels.
%A. A code has been used to check intensity levels and is shown above.


%% 3. An image displayed with two different colormaps.
% colormaps(gray) and colormaps(jet) was used on some of my images.

%% 4.MSE vs. N for noisy images
% A graph has been plotted between MSE and N values and the code has been
% implemented.

%% 5.Result of affine transformation.
% Bike.cr2 image file was used to test the rotation and scaling values. The
% image rotated successfully at an angle of 30 degrees and a scaling value
% of 0.7 was also used.