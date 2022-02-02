%% Compare the effect of iterations
% -------------------------------------------------------------------------
% clean the workspace
% -------------------------------------------------------------------------
clc
clear all
close all
warning off
rng('default')
rng(1) % fix seed
addpath(genpath("."));

% -------------------------------------------------------------------------
% Simulation setting
% -------------------------------------------------------------------------
B = 100;                              % replication number
p = 5000;                             % Signal dimension
N = 21600;                            % sample size
n = 720;                              % local sample size
L = N / n;                            % node number
K = 30;                               % signal sparsity level
rflipmodel = [1/4; 1/8];              % the probabilities of sign-flips
sigmamodel = [0.1; 0.2];              % the noise intensity
index = int16(rand(L, 1)<0.5)+1;      % index for the first model
rflip = rflipmodel(index);
sigma = sigmamodel(index);
rho = 0.5;                            % coorelation
Maxiter_dist = 10;                    % maximum number of distributed iterations


% -------------------------------------------------------------------------
% Metrics
% -------------------------------------------------------------------------
error = zeros((4 + Maxiter_dist), 1); % \ell_2 error
f1 = zeros((4 + Maxiter_dist), 1);    % F_1 score

% -------------------------------------------------------------------------
% Estimate the true signal
% -------------------------------------------------------------------------
tic
for jj = 1:B
    rng(jj);                          % for replication
    if(~mod(jj,10))
        jj
    end

    [X, y, betaT, supp, Xaug, yaug, tau2] = dataGen(N, L,  'p', p,  ...
        'K',  K,  'rflip', rflip,  ...
        'sigma', sigma,  'rho', rho); % generate data

    % Local
    betaLocal = lasso(X(1:n, :), y(1:n));
    betaLocal = betaLocal / norm(betaLocal);
    error(1) = error(1) + computeError(betaLocal, betaT);
    esupp = find(betaLocal);
    f1(1) = f1(1) + computeF1(esupp, supp);

    % Avg-DC
    betaAvg = ones(p, L);
    for ii = 1:L
        betaAvg(:, ii) = lasso(X((1 + (ii - 1) * n):(ii * n), :), ...
            y((1 + (ii - 1) * n):(ii * n), :));
    end

    betaAvg = mean(betaAvg, 2);
    betaAvg = betaAvg / norm(betaAvg);
    error(2) = error(2) + computeError(betaAvg, betaT);
    esupp = find(betaAvg);
    f1(2) = f1(2) + computeF1(esupp, supp);

    % Global
    betaGlobal = lasso(X, y);
    betaGlobal = betaGlobal / norm(betaGlobal);
    error(3) = error(3) + computeError(betaGlobal, betaT);
    esupp = find(betaGlobal);
    f1(3) = f1(3) + computeF1(esupp, supp);

    % Distributed
    [betaDistributed, ithist] = lassoDistributed(X, y, L,  'Maxiter_dist', 10);
    betaDistributed = betaDistributed / norm(betaDistributed);
    ithist.beta = num2cell(ithist.beta, 1);
    ithist.beta = cellfun(@(x)(x / norm(x)), ithist.beta,  'UniformOutput', 0);
    ithist.error = cellfun(@(x)(computeError(x, betaT)), ithist.beta);
    ithist.f1 = cellfun(@(x)(computeF1(find(x), supp)), ithist.beta);
    error(5:end) = error(5:end) + ithist.error(:);
    f1(5:end) = f1(5:end) + ithist.f1(:);

    % KSW2016
    thre = 0.001;
    [~, beta_KSW, normxEst] = normEstPV_Alternate(Xaug,yaug, K, tau2);
    beta_KSW = beta_KSW.*(abs(beta_KSW)>thre);
    beta_KSW = beta_KSW/norm(beta_KSW);
    esupp = find(beta_KSW);
    error(4) = error(4) + computeError(beta_KSW, betaT);
    f1(4) = f1(4) + computeF1(esupp, supp);

end % Elapsed time is 3283.874668 seconds.

error = error / B;
f1 = f1 / B;
toc

clear X y;
save("output/compare_iterations-"+ ...
    char(datetime(now,  'ConvertFrom',  'datenum',  'Format',  ...
    'yyyy-MM-dd-HH-mm-ss')) +  ".mat");
%% plot
Markers = 10; % Marker size
FontSize = 12; % Font size
FontSize_title = 16; % Font size for the title
fig1 = figure('DefaultAxesFontSize',FontSize); % figure 1
plot(1:Maxiter_dist, error(5:end),  '*-', 'color', 	...
    '#7E2F8E', 'LineWidth', 1, 'MarkerSize', Markers)
hold on
plot(1:Maxiter_dist, error(4) * ones(Maxiter_dist, 1),  's--', 'color', 	...
    '#EDB120', 'LineWidth', 1, 'MarkerSize', Markers)
plot(1:Maxiter_dist, error(3) * ones(Maxiter_dist, 1),  'd:', 'color', 	...
    '#D95319', 'LineWidth', 1, 'MarkerSize', Markers)
plot(1:Maxiter_dist, error(2) * ones(Maxiter_dist, 1),  'v-.', 'color', 	...
    '#0072BD', 'LineWidth', 1, 'MarkerSize', Markers)
plot(1:Maxiter_dist, error(1) * ones(Maxiter_dist, 1),  'o-.', 'color', 	...
    '#77AC30', 'LineWidth', 1, 'MarkerSize', Markers)
lgd = legend({'distributed',  'KSW', 'pooled',  'dc',  'local'}, ...
    'Position',[0.689672619047619, 0.576666666666667, ...
    0.175892857142857 0.186904761904762]);
lgd.FontSize = FontSize;
hold off
t = title('$\ell_2$-error',  'Interpreter',  'latex');
t.FontSize = FontSize_title;
print(fig1,  '-depsc',  'fig/iterations_error_heter.eps');


fig2 = figure('DefaultAxesFontSize',FontSize);
plot(1:Maxiter_dist, f1(5:end),  '*-', 'color', 	...
    '#7E2F8E', 'LineWidth', 1, 'MarkerSize', Markers)
hold on
plot(1:Maxiter_dist, f1(4) * ones(Maxiter_dist, 1),  's--', 'color', 	...
    '#EDB120', 'LineWidth', 1, 'MarkerSize', Markers)
plot(1:Maxiter_dist, f1(3) * ones(Maxiter_dist, 1),  'd:', 'color', 	...
    '#D95319', 'LineWidth', 1, 'MarkerSize', Markers)
plot(1:Maxiter_dist, f1(2) * ones(Maxiter_dist, 1),  'v-.', 'color', 	...
    '#0072BD', 'LineWidth', 1, 'MarkerSize', Markers)
plot(1:Maxiter_dist, f1(1) * ones(Maxiter_dist, 1),  'o-.', 'color', 	...
    '#77AC30', 'LineWidth', 1, 'MarkerSize', Markers)
lgd = legend('distributed',  'KSW', 'pooled',  'dc',  'local', ...
    'Position',[0.689672619047619, 0.297857142857143, ...
    0.175892857142857, 0.186904761904762]);
lgd.FontSize = FontSize;
hold off
t = title('$F_1$-score ',  'Interpreter',  'latex');
t.FontSize = FontSize_title;
print(fig2,  '-depsc',  'fig/iterations_f1_heter.eps');
