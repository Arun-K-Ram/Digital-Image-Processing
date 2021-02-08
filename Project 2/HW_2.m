%% Project 2:   Intensity Transformations
%% Course No:   ECE 5256
%% Due Date:    02/07/2021
%% Q1.) Image Enhancement Using Intensity Transformations
%The focus here is to experiment with intensity transformations to enhance an image. Download an image and enhance it using the transformation, s = crv
%There are two parameters, c and r for which values have to be selected. As in most enhancement tasks, experimentation is a must. The objective of this project is to obtain the best visual enhancement possible. Once (according to your judgment) you have the best visual result, and explain the reasons for your choice.
%% Read an Image
I=imread('Forest.jpg');
%The MATLAB function that creates these gamma transformations is:
%imadjust(f, [low_in high_in], [low_out high_out], gamma)
%f=imput image,gamma=controls curve,[low_in high_in] and [low_out high_out]
%are used for clipping

J=imadjust(I,[],[],1);
J2=imadjust(I,[],[],3);
J3=imadjust(I,[],[],0.4);
J4=imadjust(I,[],[],2.5);
J5=imadjust(I,[],[],1.8);
imshow(J);
title("Gamma = 1")
figure,imshow(J2);
title("Gamma = 3")
figure,imshow(J3);
title("Gamma = 0.4")
figure,imshow(J4);
title("Gamma = 2.5")
figure,imshow(J5);
title("Gamma = 1.8")

%% Result: The following shows the results of five of the gamma transformations 
%shown in the plot above. We can see that the values greater than 1 one create a darker image,
%whereas values between 0 and 1 create a brighter image with more contrast in 
%dark areas so that you can see the details of the forest.
%% Q2.) Histogram Equalization
%% a) Display an Image
Z=imread('Sea.jpg');
imshow(Z)
%% b) Plot its histogram
numofpixels=size(Z,1)*size(Z,2);
figure,histogram(Z);
%% C)Plot the histogram-equalization transformation function (cumulative distribution probability is calculated). 
im = imread('Farm.jpg');
im_hist = imhist(im);

%transformation function
tf = cumsum(im_hist); 
tf_norm = tf / max(tf);
plot(tf_norm), axis tight

% obtain CDF of the image.   
[histIM, bins] = imhist(im);
[counts, bins] = imhist(im);
cdf = cumsum(counts) / sum(counts);
plot(cdf)
title("CDF of the Image");
figure,histogram(cdf)
%% d) Histogram-equalized Image

Z=imread('Sea.jpg');
HIm=uint8(zeros(size(Z,1),size(Z,2)));

freq=zeros(256,1);

probf=zeros(256,1);

probc=zeros(256,1);

cum=zeros(256,1);

output=zeros(256,1);


%freq counts the occurrence of each pixel value.

%The probability of each occurrence is calculated by probf.


for i=1:size(Z,1)

    for j=1:size(Z,2)

        value=Z(i,j);

        freq(value+1)=freq(value+1)+1;

        probf(value+1)=freq(value+1)/numofpixels;

    end

end
sum=0;
no_bins=255;
for i=1:size(probf)

   sum=sum+freq(i);

   cum(i)=sum;

   probc(i)=cum(i)/numofpixels;

   output(i)=round(probc(i)*no_bins);

end

for i=1:size(Z,1)

    for j=1:size(Z,2)

            HIm(i,j)=output(Z(i,j)+1);

    end

end
figure,imshow(HIm);

title('Histogram equalization');

%% e) plot the histogram-equalized image histogram
title("histogram-equalized image histogram")
histogram(HIm);
%% 3) Spatial filtering
% a) Add noise to an image whose maximum intensity is close to 255 on 
%a scale of 0-255 using 10*randn (standard deviation of 10, mean of 0).
M = imread("Farm.jpg");

% Adding max intensity to an image
M_M = mat2gray(M,[0,255]); 

% Adding noise to image
%noisy_img = imnoise(M_M,'gaussian',0,100);
noisy_img = M + (10.*randn);
title("Noisy Image");
imshow(noisy_img);
%% b) Use a Sobel edge detector to display the edges in an image.
M = imread("Farm.jpg");
N=rgb2gray(M);

C=double(N);


for i=1:size(C,1)-2
    for j=1:size(C,2)-2
        %Sobel mask for x-direction:
        Gx=((2*C(i+2,j+1)+C(i+2,j)+C(i+2,j+2))-(2*C(i,j+1)+C(i,j)+C(i,j+2)));
        %Sobel mask for y-direction:
        Gy=((2*C(i+1,j+2)+C(i,j+2)+C(i+2,j+2))-(2*C(i+1,j)+C(i,j)+C(i+2,j)));
     
        %The gradient of the image
        %B(i,j)=abs(Gx)+abs(Gy);
        N(i,j)=sqrt(Gx.^2+Gy.^2);
     
    end
end
figure,imshow(N); title('Sobel gradient');
%% c) Low-pass filter the original image before detecting edges using a
%two different filters in an attempt to obtain smooth edges.

%A low pass filter is the basis for most smoothing methods.
%An image is smoothed by decreasing the disparity between pixel values by averaging nearby pixels (see Smoothing an Image for more information).

% The low pass frequency components denotes smooth regions.
%Using a low pass filter tends to retain the low frequency information within an image while reducing the high frequency information
if true
  % code
clc
close all
clear all
I1 = imread('Sea.jpg');
I1 = imresize(I1,[128 128]);
I2 = rgb2gray(I1)
I2=double(I2);
figure, imshow(uint8(I2));
I3=fft2(I2);
I3=fftshift(I3);
figure
I4=log(1+abs(I3));
imshow(mat2gray(I4));
[r,c]=size(I2);
orgr=r/2;
orgc=c/2;
mf= zeros(r,c);
D0= 40;
for i=1:r
  for j=1:c
      if((i-orgr)^2+(j-orgc)^2)^(0.5)<=D0
          mf(i,j)=1;
      end
  end
end
figure
imshow(uint8(255*mf));
title('frequency domain filter used');
I5=I3.*mf;
figure,
I4=log(1+abs(I5));
imshow(mat2gray(I4));
title('filtered image in frequency domain');
I6=ifft2(ifftshift(I5));
figure,
imshow(uint8(abs(I6)));
title('filtered gray scale image');
end
%% Create a blur image using Gaussian Low pass filter
% Create a blurring kernel.
img = imread("Forest.jpg");
kernel = fspecial('Gaussian', 32, 8);
subplot(2, 2, 2);
imshow(kernel, []); 
axis on;
title('Blurring Kernel');
% Blur the image.
blurred = imfilter(img, kernel, 'replicate');
subplot(2, 2, 3); 
imshow(blurred); 
axis on;
title('Blurred Image');
%% Q 3.2)  Done and attached separately
%% Q3.7) Done and attached separately. 