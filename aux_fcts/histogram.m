function [h,bins,x_bins] = histogram(x,num_bins)

x_max = max(abs(x(:)));
x_bins = 0:x_max/num_bins:x_max-x_max/num_bins;
[h,bins] = hist(abs(x(:)),x_bins);

end