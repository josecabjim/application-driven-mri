function [mask, acc] = var_dens_mask(Nx,Ny,Nt,ivar)

pdf_x = pdf_gauss(Nx,ivar).';
pdf_y = pdf_gauss(Ny,ivar).';
pdf = pdf_x'*pdf_y;
pdf = pdf/1.1 + 0.01;
for t=1:Nt
    mask(:,:,t) = binornd(1, pdf);
    mask(round(Nx/2-5:Nx/2+5),round(Ny/2-5:Ny/2+5),t) = true;
end
mask_batch(:,:,:) = mask;
acc = numel(mask_batch(:,:,:))/sum(sum(sum(mask_batch(:,:,:))));

end