function [x_out, AnalysisCode] = OMPcode(x, D, paramKSVD)
% [x_out, AnalysisCode] = OMPcode(x, D, paramKSVD)
% 
% Sparsely code an image with OMP.
%
% INPUTS
%   x...                         Image to code
%   paramKSVD.
%      blocksize...              Patch and atom size (only 1 dim)
%      dictsize...               Number of dictionary atoms
%      iternum...                Number of KSVD iterations (if = 0, no training)
%      trainnum...               Number of training patches
%      sigma...                  Maximum error allowed in patch
%                                representation (notice gain *1.15 below)
%      maxatoms...               Maximum number of atoms used per patch
%      analyse...                if = 1 will output useful analysis data
%
% OUTPUTS
%   x_out...                     Coded image
%   AnalysisCode...              Analysis data output (used if paramKSVD.analyse=1)
%      error...                  MSE produced by sparse coding
%      coefs...                  Mean atoms used per patch

%  Jose Caballero
%  Department of Computing
%  Imperial College London
%  jose.caballero06@gmail.com
%
%  May 2014


paramKSVD.dict = D;
paramKSVD.lambda = 0;

if isreal(x)
    fprintf(' --- OMP code:\n');
    paramKSVD.x = x;
    if ismatrix(x); [x_out,AnalysisCode.coefs] = ompdenoise2(paramKSVD,5);
    else [x_out,AnalysisCode.coefs] = ompdenoise3(paramKSVD,5);
    end
else
    fprintf(' --- OMP code: Real part\n');
    paramKSVD.x = real(x);
    if ismatrix(x); [x_out_r,AnalysisCode.coefs_r] = ompdenoise2(paramKSVD,5);
    else [x_out_r,AnalysisCode.coefs_r] = ompdenoise3(paramKSVD,5);
    end
    fprintf('\n --- OMP code: Imaginary part\n');
    paramKSVD.x = imag(x);
    if ismatrix(x); [x_out_i,AnalysisCode.coefs_i] = ompdenoise2(paramKSVD,5);
    else [x_out_i,AnalysisCode.coefs_i] = ompdenoise3(paramKSVD,5);
    end
    x_out = x_out_r + 1i*x_out_i;
    AnalysisCode.coefs = AnalysisCode.coefs_r + AnalysisCode.coefs_i;
end

AnalysisCode.error = sqrt(mean(abs(x_out(:)-x(:)).^2));

end