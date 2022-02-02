function [beta, ithist] = lassoDistributed(X,y,L,varargin)
%LASSODISTRIBUTED Perform lasso regularization for linear regression with
%  data scattered across L nodes.
%  [beta,esupp] = lassoDistributed(X,y,varargin) Performs L1-constrained linear least
%  squares fits (lasso) relating the predictors in X to the responses in y.
%  The default is a lasso fit, or constraint on the L1-norm of the
%  coefficients B.
%
%  Positional parameters:
%
%    X                A numeric matrix (dimension, say, Nxp)
%    y                A numeric vector of length N
%    L                A numeric scalar, the number of nodes.
%
%  Optional input parameters:
%    Maxiter_dist     The maximum times of iterations
%    init_dist        The initial estimator for the true signal beta
%    parallel         Whether to parallel (false by default).
%    tol              Tolerance for stopping (1e-4 by default).
%  Return values:
%    beta             The decoded signal.
%    ithist           The history of iterations.
%  Examples:
%
%  See also .

%  References:
%
%

% todo: parallelized
% --------------------------------------
% Sanity check the positional parameters
% --------------------------------------

[N,p] = size(X);
n = N/L;
pnames = {'Maxiter_dist' 'init_dist' 'parallel' 'tol'};
dflts = {10 zeros(p,1) false 1e-4};
[Maxiter_dist, init_dist, parallel, tol] = parseArgs(pnames, dflts, varargin{:});


if parallel
    % --------------------------------------
    % Initials
    % --------------------------------------
    ithist.beta = [];
    bSig1 = X(1:n,:)'*X(1:n,:)/n;
    
    % --------------------------------------
    % Distributed estimate
    % --------------------------------------
    b = bSig1*init_dist;
    Xbetaarr = zeros(p,L);
    zNarr = zeros(p,L);
    parfor i = 1:L
        Xbetaarr(:,i) = X((1 + (i - 1)*n):i*n,:)'*X((1 + (i - 1)*n):i*n,:) ...
            /N*init_dist;
        zNarr(:,i) = X((1 + (i - 1)*n):L*n,:)'*y((1 + (i - 1)*n):L*n)/N;
    end
    zN = sum(zNarr, 2);
    b = b + zN - sum(Xbetaarr, 2);
    
    beta = pdasc(bSig1,b,N,p); % should change to N
    ithist.beta(:,1) = beta;
    
    for ii = 2:Maxiter_dist
        b = zN + bSig1*ithist.beta(:,(ii-1));
        parfor i = 1:L
            Xbetaarr(:,i) = X((1 + (i - 1)*n):i*n,:)'* ...
                X((1 + (i - 1)*n):i*n,:)/N*ithist.beta(:,(ii-1));
        end
        b = b - sum(Xbetaarr, 2);
        
        beta = pdasc(bSig1,b,N,p); % should change to N
        ithist.beta(:,ii) = beta;
        if norm(ithist.beta(:,ii)/norm(ithist.beta(:,ii)) - ...
                ithist.beta(:,ii-1)/norm(ithist.beta(:,ii-1)))<=tol
            ii
            break
        end
    end
else
    % --------------------------------------
    % Initials
    % --------------------------------------
    
    ithist.beta = [];
    
    bSig = X'*X/N;
    bSig1 = X(1:n,:)'*X(1:n,:)/n;
    zN = X'*y/N;
    
    % --------------------------------------
    % Distributed estimate
    % --------------------------------------
    
    % ii = 1
    b = zN + (bSig1-bSig)*init_dist;
    beta = pdasc(bSig1,b,N,p); % 这里应该改成n
    ithist.beta(:,1) = beta;
    
    for ii = 2:Maxiter_dist
        b = zN + (bSig1-bSig)*ithist.beta(:,(ii-1));
        beta = pdasc(bSig1,b,N,p); % 这里应该改成n
        ithist.beta(:,ii) = beta;
    end
end %-parallel
end %-lassoDistributed
