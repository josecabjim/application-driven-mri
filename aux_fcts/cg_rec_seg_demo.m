function [X_recon, flag_cg] = cg_rec_seg_demo(mask,X_acq,X_D,X_init,gmm,tun,flag_gmm_term)
% CG_REC_SEG_DEMO
%
% Function solving the following optimisation problem wrt x with CG:
%
% min_x
%  |MFx-y|^2 + \lambda/Np*sum_i[|R*x_i-D*\gamma_i|^2] - \beta*P(x|mu,sig,pi_k)
%    s.t.  |gamma_i|_0 <= T
%
% Inputs:
%   mask...             K-space undersampling mask
%   X_acq...            Acquired K-space samples
%   X_D...              K-space representation of data approximated by sparse dictionary
%   X_init...           CG initial estimate
%   gmm.                Initial Gaussian Mixture Model (gmm)
%       mu...           Gaussian means
%       sig...          Gaussian standard deviations
%       pi_k...         Gaussian mixture coefficients  
%   tun.                Tuning parameters
%       lambda...           Tuning parameter for dictionary term
%       beta...             Tuning parameter for GMM term
%   flag_gmm_term...    Flag deciding whether GMM term should be considered
%                       (if flag_gmm_term = 0, then force beta = 0)
%
% Outputs
%  X_recon...           K-space result for image x
%  flag_cg...           CG algorithm flag with information on convergence

%  Jose Caballero
%  Department of Computing
%  Imperial College London
%  jose.caballero06@gmail.com
%
%  September 2014


if flag_gmm_term==0;
    tun.beta = 0;
end

N = numel(mask);

% Initial solution
x_init = X_init;

% Calculate contribution from GMM
r = reshape(gmm.resp,size(mask,1)*size(mask,2)*size(mask,3),numel(gmm.mu));
r_sig = sum(bsxfun(@rdivide, r, gmm.sig.^2),2);
r_mu_sig = sum(bsxfun(@times, r, bsxfun(@rdivide,gmm.mu,gmm.sig.^2)),2);
r_sig = reshape(r_sig,size(mask));
R_mu_sig = my_fft2(reshape(r_mu_sig,size(mask)));

% LS solution
b = [X_acq(:); sqrt(tun.lambda)*X_D(:); sqrt(tun.beta/2)*R_mu_sig(:)];
E = @(x,tflag) aprod(x,mask,r_sig,tun,tflag);
fprintf(' - Reconstruction... ')
[tmpres,flag_cg,RELRES,ITER,RESVEC] = lsqr(E,b,1e-6,100,speye(N,N),speye(N,N),x_init(:));
X_recon = reshape(tmpres,Nx,Ny,Nt);
x_recon = abs(my_ifft2(X_recon));

idx_acq = find(mask==1);

N = bsxfun(@times,gmm.pi_k,bsxfun(@rdivide,exp(-0.5*bsxfun(@rdivide,(bsxfun(@minus, x_recon(:), gmm.mu)).^2,abs(gmm.sig).^2)),sqrt(2*pi)*gmm.sig));

function [res,tflag] = aprod(x,mask,r_sig,tun,tflag)
	
    [Nx,Ny,Nt] = size(mask);
    N = numel(mask);
    idx_acq = find(mask==1);

	if strcmp(tflag,'transp');
		x_acq = x(1:length(idx_acq));
        res1 = zeros(Nx,Ny,Nt);
        res1(idx_acq) = x_acq;

        x_D_k = x((length(idx_acq)+1):(length(idx_acq)+Nx*Ny*Nt));
        res2 = reshape(x_D_k,Nx,Ny,Nt);

        x_k_seg = x((length(idx_acq)+Nx*Ny*Nt+1):end);
        x_k_seg = reshape(x_k_seg,Nx,Ny,Nt);
        x_im_seg = my_ifft2(x_k_seg);
        res3 = my_fft2(x_im_seg.*r_sig);
        
        res = res1 + sqrt(tun.lambda)*res2 + sqrt(tun.beta/2)*res3;
        res = res(:);
	
    else
        x_mat = reshape(x,Nx,Ny,Nt);
        
        res1 = x_mat(idx_acq);
        
        res2 = x_mat;

        x_mat_im = my_ifft2(x_mat);
        res3 = my_fft2(x_mat_im.*r_sig);
                    
		res = [res1(:); sqrt(tun.lambda)*res2(:); sqrt(tun.beta/2)*res3(:)];
    end
end

end