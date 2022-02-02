% -------------------------------------------------------------------------
% Clean the workspace
% -------------------------------------------------------------------------
clc
clear all
close all
warning off
rng('default')
rng(1)             % fix seed
addpath(genpath('simulations')); addpath(genpath('matlab'));
% -------------------------------------------------------------------------
% Simulation setting
% -------------------------------------------------------------------------
nClusers = 50;     % the number of workers
myCluster=parcluster('local');
myCluster.NumWorkers=nClusers;
parpool(myCluster,nClusers)

B = 100;           % replication number
p = 5000;          % Signal dimension
K = 30;            % signal sparsity level
rho = 0.5;         % coorelation
Maxiter_dist = 10; % maximum number of distributed iterations

% settings
sigma_max = (1:6);
NN = [21600]; % sample size
args = [reshape(repmat(sigma_max, length(NN), 1), [], 1), ...
    reshape(repmat(NN, 1, length(sigma_max)), [], 1)];
args1 = args(:, 1);
args2 = args(:, 2);

if any(args1 < 2 * K * log(p))
    disp("The local sample size is too small!")
end

% -------------------------------------------------------------------------
% Metrics
% -------------------------------------------------------------------------
M = 5;
avg_err = zeros(M, length(sigma_max) * length(NN));
avg_f1 = zeros(M, length(sigma_max) * length(NN));
avg_precision = zeros(M, length(sigma_max) * length(NN));
avg_recall = zeros(M, length(sigma_max) * length(NN));

% -------------------------------------------------------------------------
% Estimate the true signal
% -------------------------------------------------------------------------
tic
for i = 1:length(sigma_max) * length(NN)
    i
    n = 1440;           % local sample size
    N = args2(i);
    L = N / n;
    sigmaM = 0.2 * args1(i);
    rflipmodel = [1/4; 1/8];
    sigmamodel = [0.1; sigmaM];


    % metrics
    error = zeros(M, 1);
    f1 = zeros(M, 1);
    precision = zeros(M, 1);
    recall = zeros(M, 1);
    parfor jj = 1:B
        rng(jj)         % replication
        if(~mod(jj,10))
            jj
        end
        index = int16(rand(L, 1) < 0.5) + 1;
        rflip = rflipmodel(index);
        sigma = sigmamodel(index);
        [X, y, betaT, supp, Xaug, yaug, tau2] = dataGen(N, L, 'p', ...
            p, 'K', K, 'rflip', rflip, 'sigma', sigma, 'rho', rho);
        [betaE, jjerror, jjf1, ...
            jjprecision, jjrecall] = fit(X, y, n, betaT, Xaug,yaug, K, tau2);
        error = error + jjerror;
        f1 = f1 + jjf1;
        precision = precision +jjprecision;
        recall = recall + jjrecall;
    end

    error = error / B;
    f1 = f1 / B;
    precision = precision / B;
    recall = recall / B;

    avg_err(:, i) = error;
    avg_f1(:, i) = f1;
    avg_precision(:, i) = precision;
    avg_recall(:, i) = recall;

end
toc
avg_err
avg_f1

delete(gcp('nocreate'))

%% output the latex table
avg_err_f1 = zeros(length(args1), M*2 + 2);
avg_err_f1(:, (1:M) * 2 + 2) = avg_err';
avg_err_f1(:, (1:M) * 2 + 1) = avg_f1';
avg_err_f1(:, 1) = args1;
avg_err_f1(:, 2) = args2;
% for latexTable input
input.data = avg_err_f1;
input.tableColLabels = repmat({'$F_1$-score', '$\ell_2$-error$'}, 1, M+1);
input.tableCaption = '';                   % Caption
input.dataFormat = {'%d', 2, '%.4f', M*2}; % format
input.tableBorders = 0;                    % turn off table borders
input.makeCompleteLatexDocument = 0;
latex = latexTable(input);

dlmwrite("output/compare_heterogeneity_sigma-" + ...
    char(datetime(now, 'ConvertFrom', 'datenum', ...
    'Format', 'yyyy-MM-dd-HH-mm-ss')) + ".txt", ...
    char(latex), 'delimiter', '')          % write to file
clear X y;
save("output/compare_heterogeneity_sigma-" + ...
    char(datetime(now, 'ConvertFrom', 'datenum', ...
    'Format', 'yyyy-MM-dd-HH-mm-ss')) + ".mat");

%% plot
Markers = 10;
FontSize = 12;
FontSize_title = 16;
fig1 = figure('DefaultAxesFontSize',FontSize);
plot(sigma_max, avg_err(2, :), 'v-.', ...
    sigma_max, avg_err(3, :), 'd:', ...
    sigma_max, avg_err(4, :), 's--', ...
    sigma_max, avg_err(5, :), '*-', ...
    'LineWidth', 1, 'MarkerSize', Markers)
lgd = legend('dc', 'pooled', 'KSW', 'distributed', ...
    'Location','best','NumColumns',1);
lgd.FontSize = FontSize;
set(gca, 'XTick', sigma_max);
t = title('$\ell_2$-error', 'Interpreter', 'latex');
t.FontSize = FontSize_title;
print(fig1, '-depsc', 'fig/effect_of_heter_sigma_error.eps');

fig2 = figure('DefaultAxesFontSize',FontSize);
plot(sigma_max, avg_f1(2, :), 'v-.', ...
    sigma_max, avg_f1(3, :), 'd:', ...
    sigma_max, avg_f1(4, :), 's--', ...
    sigma_max, avg_f1(5, :), '*-', ...
    'LineWidth', 1, 'MarkerSize', Markers)
lgd = legend('dc', 'pooled', 'KSW', 'distributed', ...
    'Location','best','NumColumns',1);
lgd.FontSize = FontSize;
set(gca, 'XTick', sigma_max);
t = title('$F_1$-score', 'Interpreter', 'latex');
t.FontSize = FontSize_title;
print(fig2, '-depsc', 'fig/effect_of_heter_sigma_f1.eps');
