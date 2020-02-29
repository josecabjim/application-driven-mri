function x = my_ifft2(X)

for t = 1:size(X,4)
    for c = 1:size(X,3)
        x(:,:,c,t) = sqrt(size(X,1)*size(X,2)) * fftshift(ifft2(ifftshift(X(:,:,c,t))));
    end
end

end