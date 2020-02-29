function gmm = my_emgm_demo(data, mask_signal, gmm, max_it, tun)
% MY_EMGM_DEMO
%
% Function runing the EM algorithm on an image.
%
% Inputs
%   data...         Input data (image) to be segmented
%   mask_signal...  Mask selecting pixels to segment
%   gmm.            Initial Gaussian Mixture Model (gmm)
%       mu...           Gaussian means
%       sig...          Gaussian standard deviations
%       pi_k...         Gaussian mixture coefficients
%   max_it...       Maximum EM iterations
%   tun.            Tuning parameters
%       min_sig...      Forced minimum standard deviation
%
% Outputs
%   gmm.            Output Gaussian Mixture Model (gmm)
%       mu...           Gaussian means
%       sig...          Gaussian standard deviations
%       pi_k...         Gaussian mixture coefficients

%  Jose Caballero
%  Department of Computing
%  Imperial College London
%  jose.caballero06@gmail.com
%
%  September 2014

[Nx,Ny,Nt] = size(data);

data = data(mask_signal);
data = data(:);

if isstruct(gmm)  % initialize with a gmm
    K = size(gmm.mu,2);
    mu = gmm.mu;
    sig = gmm.sig;
    pi_k = gmm.pi_k;
else
    K = gmm;
    e95 = 0.95*norm(data(:));
    for e95ind=1:-0.01:0
        if norm(data(data<e95ind)) <= e95
            break
        end
    end
    mu = 0:e95ind/K:e95ind-e95ind/K; % Initialise mu evenly from 0 to 95% of signal energy
%     mu = randsample(data,K)'; % Uncomment to initialise random mu
    sig = .1 * ones(1,K);
    pi_k = ones(1,K)/K;
    clear gmm
end

for jj=1:max_it
    
    % E-step
    N = bsxfun(@rdivide,exp(-0.5*bsxfun(@rdivide,(bsxfun(@minus, data, mu)).^2,abs(sig).^2)),sqrt(2*pi)*sig);
    
    pi_N = bsxfun(@times,pi_k,N);

    sum_pi_N = sum(pi_N,2);
    r_zik = bsxfun(@rdivide,pi_N,sum_pi_N);

    % M-step
    mu = sum(bsxfun(@times,r_zik,data),1) ...
        ./ sum(r_zik,1);
    [mu,order] = sort(mu,'ascend');r_zik = r_zik(:,order); % order by mu
    if sum(diff(mu)<0.05)>0;
        mu(diff(mu)<0.05) = rand;
    end
    xmmu = bsxfun(@minus, data, mu); 
    sig = sqrt( sum(r_zik .* abs(xmmu).^2, 1)./(sum(r_zik,1)) );
    pi_k = sum(r_zik,1)/numel(data);

    % Force st.d. limits
    if isfield(tun,'min_sig')
        for k=1:numel(sig);
            sig(k) = max(sig(k),tun.min_sig); % Force minimim variance
        end
    end
    
end;

gmm.mu = mu;
gmm.sig = sig;
gmm.pi_k = pi_k;

% Reformat output
gmm.resp = zeros(Nx,Ny,Nt,K);
gmm.seg = zeros(Nx,Ny,Nt,K);
for k = 1:K
    temp_r = zeros(Nx,Ny,Nt);
    temp_r(mask_signal) = r_zik(:,k);
    gmm.resp(:,:,:,k) = temp_r;
end
gmm.seg = zeros(Nx,Ny,Nt);
[~,gmm.seg(mask_signal)] = max(r_zik,[],2);

end