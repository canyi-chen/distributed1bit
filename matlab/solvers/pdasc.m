function [beta, lam, ithist] = pdasc(A, b, n, p, opts)
%PDASC Quadratic lasso solver
% f = PDASC(A, b, n, p, opts) computes the quadratic lasso optimization
%       min 1/2*beta^T*A*beta-beta^T*b  + lambda ||beta||_1
% problem by PDAS algorithm with continuation Lam = {lam_1, ..., lam_N_Lam}.
% 
% INPUTS:
%   A       input matrix (R^{p*p})
%   b       input vector (R^p)
%   opts    structure containing
%       N_Lam      length of path (default: 300)
%       Lmin    minimun in Lam (default: 1e-4)
%       mu      stop if the size of beta_{lam_k} > mu
%                   (default:min(0.5*n/log(p),sqrt(p)))
%       init    initial value for beta (default: 0)
% 
% OUTPUTS:
%   beta    recovered signal
%   lam     regularization  parameter
%   ithist  structure on iteration history, containing
%       .beta   solution path
%       .as     size of active set  on the path
%       .it     # of iteration on the path
% 
% EXAMPLE
%   clear all, clc, close all
%   p = 1000; rho = 0.2; N = 1000; K = 5;
%   SIGMA = rho.^(abs(transpose(1:p)-(1:p)));
%   Mu = zeros(1,p);
%   X = mvnrnd(Mu,SIGMA,N);
%   betaT = sign(sprandn(p,1,K/p));
%   ye = X*betaT;
%   sigma = 0.1;
%   noise = sigma*normrnd(0,1,N,1);
%   y = sign(ye + noise);
%   A = X'*X/N; b = X'*y/N;
%   [beta, lam, ithist] = pdasc(A, b, N, p);
%   error = norm(beta/norm(beta,2)-betaT)
% REFERENCE
%   Huang, J., Jiao, Y., Lu, X. & Zhu, L. (2018), ‘Robust decoding from 1-bit
%   compressive sampling with ordinary and regularized least squares’, SIAM 
%   Journal on Scientific Computing 40(4), A2062-A2086.
% Copyright: CANYI CHEN chency1997@ruc.edu.cn



linf = norm(b, inf);
if ~exist('opts', 'var')
    opts.N_Lam = 300;
    opts.mu = min(n/log(p), sqrt(p));
    opts.Lmax = 1;
    opts.Lmin = 1e-4;
    opts.init = zeros(p, 1);
    opts.p = p;
    opts.n = n;
end

% construct the homotopy path
Lam = exp(linspace(log(opts.Lmax), log(opts.Lmin), opts.N_Lam))';
Lam = Lam(2:end); % discard the first element
Lam = Lam * linf;
ithist.Lam = Lam;
% main loop for pathfolling and choosing lambda and output solution
ithist.beta = [];
ithist.as = [];
for k = 1:length(Lam)
    opts.lam = Lam(k);
    [beta, s] = pdas(A, b, opts);
    opts.init = beta;
    ithist.beta(:, k) = beta;
    ithist.as = [ithist.as; s]; % size of active set
    if s > opts.mu
        % display('# NON-ZERO IS TOO MUCH, STOP ...')
        break
    end
end
% select the solution on the path by voting
ii = find(ithist.as == mode(ithist.as));
ii = ii(end);
beta = ithist.beta(:, ii);
lam = Lam(ii);
end %-pdasc

function [beta, s] = pdas(A, b, opts)
% f = pdas(A, b, opts) computes the minimizer of                                                      
%   1/2*beta^T*A*beta-beta^T*b  + lambda ||beta||_1               
% by  one step primal-dual active set algorithm                   
%
% INPUTS:                                                                 
%   A   input matrix (R^{p*p})   
%   b   input vector (R^p)
%   opts   structure containing                                 
%       lam    regualrization paramter                              
%       init   initial guess    
% 
% OUTPUTS:                                                                
%   beta    the minimizer                                                                                                               
%   s       size of active set                                   
lam = opts.lam;
beta0 = opts.init;
p = opts.p;
% initializing ...
pd = beta0 + (b - A * beta0); % initial guesss of d
Ac = find(abs(pd) > lam);     % active set
s = length(Ac);
beta = zeros(p, 1);
dAc = lam * sign(pd(Ac));
bAc = b(Ac);
rhs = bAc - dAc;
G = A(Ac, Ac);
beta(Ac) = G \ rhs;
end %-pdas



