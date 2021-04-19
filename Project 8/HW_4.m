%% Project 9: Segmentation
%% Course No: ECE 5256
%% Due Date:  4/18/2021
%%
%% Q1.)In this project please describe your steps, and submit your code and results for the following.
% 1 Implement Otsu's optimum thresholding in a multithreshold approach to find two optimal thresholds to segment the image Cells.tf into three regions.

I = imread("Cells.tif");
imshow(I);
title("Original Image");
%%
H_I = histeq(I);
imshow(H_I);
title("Image after applying histogram equalization");
%% Determining the histogram of the Image
H = imhist(I);
% sum the values of all the histogram values
N=sum(H);
% Setting the maximum value to zero
max = 0;
%% Computing the probability of each intensity value
for i=1:256
    P(i)=H(i)/N; %Computing the probability of each intensity level
end
%% Computing a single threshold using Otsus Method
for T=2:255      % step through all thresholds from 2 to 255
    w0=sum(P(1:T)); % Probability of class 1 (separated by threshold)
    w1=sum(P(T+1:256)); %probability of class2 (separated by threshold)
    u0=dot([0:T-1],P(1:T))/w0; % class mean u0
    u1=dot([T:255],P(T+1:256))/w1; % class mean u1
    sigma=w0*w1*((u1-u0)^2); % compute sigma i.e variance(between class)
    if sigma>max % compare sigma with maximum 
        max=sigma; % update the value of max i.e max=sigma
        threshold=T-1; % desired threshold corresponds to maximum variance of between class
    end
end

%% From above, we can see that the optimum single threshold value obtained is 181.
%%
bw=im2bw(I,threshold/255); % Convert to Binary Image
figure(3),imshow(bw); % Display the Binary Image
title("Image having a single threshold");
%% Segment the image into three regions using imquantize , specifying the threshold level returned by multithresh
thresh = multithresh(I,2); %Calculate two threshold levels.
seg_I = imquantize(I,thresh); %Segment the image into three levels using imquantize .
figure;
imshow(seg_I,[])
title("Segmented Image after applying two thresholds and segmented to three regions using Otsus Algorithm");