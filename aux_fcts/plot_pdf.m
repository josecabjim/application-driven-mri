function p = plot_pdf(gmm,x,mask_signal,num_bins)
K = numel(gmm.pi_k);

x_range = 0:max(x(mask_signal))/num_bins:max(x(mask_signal))-1/num_bins;
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
    
end