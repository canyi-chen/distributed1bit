function [beta,esupp] = lasso(X,y,varargin)
%LASSO Perform lasso regularization for linear regression.
%   [beta,esupp] = lasso(X,y,varargin) Performs L1-constrained linear least  
%   squares fits (lasso) relating the predictors in X to the responses in y.
%   The default is a lasso fit, or constraint on the L1-norm of the 
%   coefficients B.
%  
%  Positional parameters:
%
%    X                A numeric matrix (dimension, say, Nxp)
%    y                A numeric vector of length N
%
%  Optional input parameters:
%
%
%  Return values:
%    beta             The decoded signal.
%    esupp            The estimated support of signal.
%  Examples:
%    
%  See also .

%  References:
%
%
    
% get the sample size N and dimension p.
[N,p] = size(X);
% transform to quadratic form.
A = transpose(X)*X/N;
b = transpose(X)*y/N;
% solve the lasso problem.
beta = pdasc(A,b,N,p);
% normalized beta
% beta = beta/norm(beta);
esupp = find(beta);
end %-lasso
