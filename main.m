% MATLAB demo 
%
% This demo script performs the joint reconstruction segmentation of a
% retrospectively undersampled phantom of the brain (BrainWeb). Details on
% the motivation, method and results can be found in the following paper:
% 
% J. Caballero, W. Bai, A. N. Price, D. Rueckert, J. V. Hajnal, 
% "Application-driven MRI: Joint reconstruction and segmentation from
% undersampled MRI data", Proceedings of the 17th International Conference
% on Medical Imaging Computing and Computer Assisted Interventions
% (MICCAI), vol. 1, pp. 106-113, Boston, MA, USA, 2014.
% 

%
% The demo simulates undersampled data y and solves the following 
% non-linear reconstruction problem:
%
% min_{x,D,mu,sig,pi_k}
%  |MFx-y|^2 + \lambda/Np*sum_i[|R*x_i-D*\gamma_i|^2] - \beta*P(x|mu,sig,pi_k)
%    s.t.  |gamma_i|_0 <= T
%
% Parameters:
%   gauss_ivar...                Inverse variance of Gaussian in variable density undersampling mask
%   K...                         Number of Gaussians in GMM
%   num_bins...                  Number of bins in histogram
%   em_iter_or...                Number of EM iterations (first iteration)
%   em_iter_rec...               Number of EM iterations (later iteration)
%   tun.                         Tuning parameters
%       lambda...                   Tuning parameter for dictionary term
%       beta...                     Tuning parameter for GMM term
%       min_sig...                  Minimum standard deviation allowed per Gaussian in GMM
%   iter_max...                  Maximum number of algorithm iterations
%   paramKSVD.                   KSVD and OMP coding algorithm parameters
%      blocksize...                 Patch and atom size (define only 1st dim)
%      dictsize...                  Number of dictionary atoms
%      iternum...                   Number of KSVD iterations (if = 0, no training)
%      trainnum...                  Number of training patches
%      sigma...                     Maximum error allowed in patch representation
%      maxatoms...                  Maximum number of atoms used per patch
%      analyse...                   if = 1 will output sparse coding analysis data
%   display...                   Flag to plot results

%  Jose Caballero
%  Department of Computing
%  Imperial College London
%  jose.caballero06@gmail.com
%
%  September 2014

%% Initialise parameters

close all;
clear all;

gauss_ivar = 0.5e-3; % Increment for more undersampling

K = 4;
num_bins = 100;
em_iter_or = 500;
em_iter_rec = 30;

tun.lambda = 1e-3;
tun.beta = 3e-9;
tun.min_sig = 3e-2;

iter_max = 50;
paramKSVD.blocksize = 8;
paramKSVD.dictsize = 200; % May be rounded up
paramKSVD.iternum = 0;
paramKSVD.trainnum = 1e5;
paramKSVD.sigma = 7e-2; % epsilon in MICCAI paper
paramKSVD.maxatoms = 8;
paramKSVD.analyse = 1;

display = 1;

%% Load data

% Brain
load('d_ph_brain99_n0.mat');
x = abs(x)/max(x(:));
mask_signal = 1:numel(x); % Mask selecting pixels to be classified

[Nx,Ny,Nt]=size(x);
X = my_fft2(x);

%% Segment original and undersample

% Classify all pixels with GMM
gmm_true = my_emgm_demo(x, mask_signal, K, em_iter_or, tun);

% Generate undersampling mask
[mask, acc] = var_dens_mask(Nx,Ny,Nt,gauss_ivar);
X_und_n = X.*mask;
x_und_n = my_ifft2(X_und_n);

if display;
    figure; imshow(abs([x(:,:,1),mask(:,:,1),x_und_n(:,:,1)]),[]); drawnow; title('Original / Mask / Zero-filled');
    display_rec_seg(x, mask_signal, num_bins, gmm_true, 2, 0);
end

%% Joint reconstruction

% Initialisation
flag_gmm_term=0;
x_rec = abs(x_und_n);
X_rec = my_fft2(x_rec);

for iter = 1:iter_max
        
    % 1) Code OMP
    % Train dictionary with KSVD (optional)
    [Dict, AnalysisTrain] = KSVDtrain(x_rec, paramKSVD);
    % Sparse code with OMP
    [x_coded_sh, AnalysisCode] = OMPcode(x_rec, Dict, paramKSVD);
    
    X_coded_sh = my_fft2(x_coded_sh);

    % 2) Estimate GMM
    % Use longer EM in first iteration, then update GMM found
    if iter == 1
        gmm_rec = my_emgm_demo(x_rec, mask_signal, K, em_iter_or, tun); 
    else
        gmm_rec = my_emgm_demo(x_rec, mask_signal, gmm_rec, em_iter_rec, tun); 
    end

    % 3) CG reconstruction
    % Incorporate GMM term after 5 iterations (if iter < 5, beta = 0)
    if iter>=5;flag_gmm_term = 1;end
    [X_rec,flag_cg] = cg_rec_seg_demo(mask,X_und_n(mask==1),X_coded_sh,X_und_n,gmm_rec,tun,flag_gmm_term);
    x_rec = abs(my_ifft2(X_rec));

    % Calculate reconstruction and segmentation errors
    rec_err(iter) = mean(abs(x_rec(:)-x(:)).^2)/mean(abs(x_rec(:)).^2); %NMSE
    seg_err(iter) = sum(abs(gmm_rec.seg(:)-gmm_true.seg(:))>0)/numel(mask_signal);

    if display;
        display_rec_seg(x_rec, mask_signal, num_bins, gmm_rec, 2, 1);
        figure(101);plot(1:numel(seg_err),seg_err,'-b.');
    end
end