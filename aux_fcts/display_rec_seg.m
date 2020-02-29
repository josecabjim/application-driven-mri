function display_rec_seg(x, mask_signal, num_bins, gmm, cases, rec)

% Histogram
x_sig = x(mask_signal);
[hist,bins,x_bins] = histogram(x_sig, num_bins);

figure(100);
   
subplot(cases,3,1+rec*3);imshow(abs(x(:,:,1)),[]);
switch rec
    case 0
        title('Fully-sampled image');
    case 1 
        title('Reconstructed intensity image (Joint)');
    case 2
        title('Reconstructed intensity image (Separate)');
end

subplot(cases,3,2+rec*3);hold off;bar(bins,hist/sum(hist));K = numel(gmm.pi_k);
switch rec
    case 0
        title('Fully-sampled histogram/PDF');
    case 1 
        title('Reconstructed histogram/PDF (Joint)');
    case 2
        title('Reconstructed histogram/PDF (Separate)');
end

% Calculate and plot PDF from GMM
x_range = 0:max(x_sig)/num_bins:max(x_sig)-1/num_bins;
for clus_ind = 1:K        
    x_ind = 0;
    for xi = x_range
        x_ind = x_ind+1;
        p(x_ind,clus_ind) = gmm.pi_k(clus_ind) * ((1/(sqrt(2*pi)*gmm.sig(clus_ind))) * ...
            exp(-(1/2)*((xi-gmm.mu(clus_ind))/gmm.sig(clus_ind))^2));
    end
end
for clus_ind = 1:K        
    hold on; plot(x_range,sum(p,2)/sum(p(:)),'-m','LineWidth',2.5);
    hold on; plot(x_range,p(:,clus_ind)/sum(p(:)),'--r','LineWidth',2);
end
axis([0 max(x(:)) 0 max(sum(p,2)/sum(p(:)))]);grid on;
subplot(cases,3,3+rec*3);imshow(gmm.seg(:,:,1),[]);
switch rec
    case 0
        title('Fully-sampled segmentation');
    case 1 
        title('Reconstructed segmentation (Joint)');
    case 2
        title('Reconstructed segmentation (Separate)');
end

end