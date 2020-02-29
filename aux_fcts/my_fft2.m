function X = my_fft2(x)

for t = 1:size(x,4)
    for c = 1:size(x,3)
        X(:,:,c,t) = 1/sqrt(size(x,1)*size(x,2)) * fftshift(fft2(ifftshift(x(:,:,c,t))));
    end
end

end