function [D, AnalysisTrain] = KSVDtrain(x, paramKSVD)
% [D, AnalysisTrain] = KSVDtrain(x, paramKSVD)
%
% Train a DCT dictionary using the K-SVD algorithm.
%
% This function requires the use of OMPbox_v10 and KSVDbox_v13 from R.
% Rubinstein (http://www.cs.technion.ac.il/~ronrubin/software.html).
%
% INPUTS
%   x...                         Image from which to extract training data
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
%   D...                         Loaded or trained dictionary
%   AnalysisTrain...             Analysis data output (used if paramKSVD.analyse=1)
%      error...                  Mean patch error produced on the training data
%      coefs...                  Mean atoms used per patch

%  Jose Caballero
%  Department of Computing
%  Imperial College London
%  jose.caballero06@gmail.com
%
%  May 2014

fprintf(' --- KSVD train:');
        
% Load dictionary
D = odctndict(paramKSVD.blocksize,paramKSVD.dictsize,ndims(x));

% Optionally train initial dictionary
if paramKSVD.iternum == 0 
    fprintf(' Use DCT dictionary (not trained)\n'); 
    AnalysisTrain = 'Dictionary not trained';
else
    paramKSVD.initdict = D;
    paramKSVD.Edata = sqrt(paramKSVD.blocksize^ndims(x)) * paramKSVD.sigma * 1.15;

    fprintf('\n')
    if isreal(x); paramKSVD.data = extractTraining(x,paramKSVD);
    else paramKSVD.data = extractTraining([real(x),imag(x)],paramKSVD);
    end
    
    if paramKSVD.analyse >= 1; verbose = 'tr';
    else verbose = '';
    end
        
    [D,Gamma,~] = ksvd(paramKSVD,verbose);
    
    if paramKSVD.analyse >= 1
        AnalysisTrain.error = sqrt( sum(sum((abs(paramKSVD.data - D*Gamma)).^2)) / numel(paramKSVD.data) ) ;
        AnalysisTrain.coefs = sum(Gamma(:)~=0)/size(paramKSVD.data,2) ;
        if paramKSVD.analyse == 2
            AnalysisTrain.CoefMat = Gamma;
        end        
    else
        AnalysisTrain = 'No analysis data';
    end
end

fprintf('\n');
end