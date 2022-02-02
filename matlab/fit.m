function [betaE, error, f1, precision, recall, time] = fit(X, y, n, ...
    betaT, Xaug, yaug, K, tau2, varargin)
%FIT Fit the N data (X, y, Xaug, yaug) scattered across L nodes, where
%  X belongs to Nxp, Xaug belongs to Nx(p+1), y is Nx1, yaug is Nx1, n is the
%  local sample size, betaT is the true signal, K is the number of nonzero 
%  position of the true signal, and tau2 is the parameter required by KSW2016.
%  [betaE, jjerror, jjf1, jjprecision, jjrecall] = fit(X, y, n, betaT, ... 
%     Xaug,yaug, K, tau2) 
%  
%  Positional parameters:
%
%    X                The covariables matrix (dimension, say, Nxp).
%    y                The response vector of length N.
%    n                The local sample size.
%    betaT            The true signal vector of length p.
%    Xaug             The augmented covariables matrix (dimension, say, Nx(p+1)).
%    yaug             The augmented response vector of length N.
%    K                The number of nonzero position of the true signal.
%    tau2             The parameter required by KSW2016.
%
%  Return values:
%    betaE            The estimated signal (px5).
%    error            The error (px5).
%    f1               The F1 score (px5).
%    precision
%    recall
%    time             The running time for each method (px5).
%
%  Examples:
%    p = 5000;             % Signal dimension
%    N = 21600;            % sample size
%    n = 720;              % local sample size
%    L = N / n;            % node number
%    K = 30;               % signal sparsity level
%    rflipmodel = [1/4; 1/8];
%    sigmamodel = [0.1; 0.2];
%    index = int16(rand(L, 1)<0.5)+1;
%    rflip = rflipmodel(index);
%    sigma = sigmamodel(index);
%    rho = 0.5;            % coorelation
%    Maxiter_dist = 10;    % maximum number of distributed iterations
%    [X, y, betaT, supp] = dataGen(N, L,  'p', p,  'K', K,  'rflip', rflip, ...
%        'sigma', sigma,  'rho', rho);   
%    [betaE, jjerror, jjf1, jjprecision, jjrecall] = fit(X, y, n, betaT, ...
%            Xaug,yaug, K, tau2);


%  References:
%
%

pnames = {'MAXITER'};
dflts = {5};
[MAXITER] = parseArgs(pnames, dflts, varargin{:});

[N, p] = size(X);
L = N/n;
supp = find(betaT);

M = 5;
error = zeros(M,1);
f1 = zeros(M,1);
precision = zeros(M,1);
recall = zeros(M,1);
time = zeros(M,1);

% Local
tic
[betaLocal, esupp] = lasso(X(1:n,:),y(1:n));
betaLocal = force_first_positive(betaLocal);
betaLocal = betaLocal/norm(betaLocal);
time(1) = toc;
error(1) = computeError(betaLocal, betaT);
[if1, iprecision, irecall] = computeF1(esupp, supp);
f1(1) = if1;
precision(1) = iprecision; % precision
recall(1) = irecall;       % recall

% Avg-DC
tic
betaAvg = ones(p,L);
for ii = 1:L
    betaAvg(:,ii) = lasso(X((1+(ii-1)*n):(ii*n),:),y((1+(ii-1)*n):(ii*n),:));
    betaAvg(:,ii) = force_first_positive(betaAvg(:,ii));
end
betaAvg = mean(betaAvg,2);
betaAvg = betaAvg/norm(betaAvg);
time(2) = toc;
error(2) = computeError(betaAvg, betaT);
esupp = find(betaAvg);
[if1, iprecision, irecall] = computeF1(esupp, supp);
f1(2) = if1;
precision(2) = iprecision; % precision
recall(2) = irecall;       % recall

% Global
tic
[betaGlobal, esupp] = lasso(X,y);
betaGlobal = force_first_positive(betaGlobal);
betaGlobal = betaGlobal/norm(betaGlobal);
time(3) = toc;
error(3) = computeError(betaGlobal, betaT);
[if1, iprecision, irecall] = computeF1(esupp, supp);
f1(3) = if1;
precision(3) = iprecision; % precision
recall(3) = irecall;       % recall

% KSW
tic
thre = 0.001;
[~, beta_KSW, ~] = normEstPV_Alternate(Xaug,yaug, K, tau2);
beta_KSW = beta_KSW.*(abs(beta_KSW)>thre);
beta_KSW = force_first_positive(beta_KSW);
beta_KSW = beta_KSW/norm(beta_KSW);
time(4) = toc;
error(4) = computeError(beta_KSW, betaT);
esupp = find(beta_KSW);
[if1, iprecision, irecall] = computeF1(esupp, supp);
f1(4) = if1;
precision(4) = iprecision; % precision
recall(4) = irecall;       % recall

% Distributed
tic
[betaDistributed, ~] = lassoDistributed(X,y,L,'Maxiter_dist', MAXITER);
betaDistributed = force_first_positive(betaDistributed);
betaDistributed = betaDistributed/norm(betaDistributed);
time(5)  = toc;
error(5) = computeError(betaDistributed, betaT);
esupp = find(betaDistributed);
[if1, iprecision, irecall] = computeF1(esupp, supp);
f1(5) = if1;
precision(5) = iprecision;                                      % precision
recall(5) = irecall;                                            % recall
betaE = [betaLocal betaAvg betaGlobal beta_KSW betaDistributed];
end