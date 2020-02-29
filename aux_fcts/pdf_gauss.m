function pdf = pdf_gauss(size_im,var)
%test file for plotting a gaussian
x=zeros(size_im,1); %initializes a row vector with length 100 of all zeros
y=zeros(size_im,1); %initializes a row vector with length 100 of all zeros 
count=0; %variable for iteration use
for i=-size_im/2:1:size_im/2-1 % i am starting a loop from -5 to +5 incrementing by 0.1 every time
count=count+1; %incrementing the iterator by 1 
x(count)=i; %storing the x value in x vector for plotting x axis
y(count)=(1)*(exp(-var*(power(i,2)))); %storing y value applying gaussian
end % ending the loop
% plot (x,y) %plotting the x and y
pdf = y;
end