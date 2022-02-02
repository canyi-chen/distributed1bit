function [X, y, betaT, supp, Xaug, yaug, tau2] = dataGen(N, L, varargin)
%DATAGEN Generate the N data scattered across L nodes.
%  [X, y, betaT, supp] = DATAGEN(N, L, ...) generate heterogenous data 
%  scattered across L nodes.
%  
%  Positional parameters:
%
%    N                A numeric scalar, the total sample size
%    L                A numeric scalar, the number of nodes
%
%  Optional input parameters:
%
%    'p'              The dimension of true signal.
%    'K'              The number of nonzero position of the true signal.
%    'rflip'          The probability of sign flips in each node. It must be
%                     a numeric vector of length L.
%    'sigma'          The variances of noises in each node. It must be a
%                     numeric vector of length L.
%    'rho'            The parameter of the covariance matrix of X. It must be
%                     a numeric scalar ranging from 0 to 1.
%  Return values:
%    X                The covariables matrix (dimension, say, Nxp).
%    y                The response vector of length N.
%    betaT            The true signal vector of length p.
%    supp             The support set of betaT.
%    Xaug             The augmented covariables matrix (dimension, say, Nx(p+1)).
%    yaug             The augmented response vector of length N.
%    tau2             The parameter required by KSW2016.
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
%  See also .

%  References:
%
%
    

% --------------------------------------
% Sanity check the positional parameters
% --------------------------------------
n = N/L;
if round(n)~=n
    error("The total sample size N is not equal to the local sample" + ...
    "size n times the number of nodes L.")
end

pnames = {'p' 'K' 'rflip' 'sigma' 'rho'};
dflts = {1000 5 (0.002).^(1:L) 0.1.^(1:L) 0.1};
[p, K, rflip, sigma, rho] = parseArgs(pnames, dflts, varargin{:});

if n<2*K*log(p)
    disp("The local sample size n = " + n + " < " + floor(K*log(p)) + ...
        " is too small.")
end

% covert to column vector
sigma = sigma(:);
rho = rho(:);

% --------------------------------------
% Generate data
% --------------------------------------

% true parameters
betaT = sign(sprandn(p,1,K/p));
betaT = force_first_positive(betaT);
% normalized true parameters
betaT = betaT/norm(betaT);
supp = find(betaT);
SIGMA = rho.^(abs(transpose(1:p)-(1:p)));
Mu = zeros(1,p);
X = mvnrnd(Mu,SIGMA,N);
ye = X*betaT;

Xaug = [X, randn(N,1)];
Rmax = 1; %upper bound for ||x||
Rmin = .1; %lower bound for ||x||

tau =Rmin; % Threshold parameter: an alternative is 
tau2 = Rmin/2+Rmax/2;  %(works better with convex minimizaiton);

% noise
noise = randn(N,1);
sigma = sigma(:,ones(1,n)); % equivalent to repmat but faster
sigma = sigma(:);
noise = sigma.*noise;

y = sign(ye + noise);
yaug = sign(Xaug*[betaT; tau2] + noise);
% todo: vectorized form
Nflip = floor(rflip*n);
for ii = 1:L
    indxflip = randperm(n);
    indxflip = indxflip(1:Nflip(ii));
    indxflip = indxflip + n*(ii-1);
    y(indxflip) = -y(indxflip);
    yaug(indxflip) = -yaug(indxflip);
end

end %-dataGen
