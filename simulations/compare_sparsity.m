%% Compare the effect of sparsity
% -------------------------------------------------------------------------
% Clean the workspace
% -------------------------------------------------------------------------
clc
clear all
close all
warning off
rng('default')
rng(1) % fix seed
addpath(genpath('simulations'));addpath(genpath('matlab'));

% -------------------------------------------------------------------------
% Simulation setting
% -------------------------------------------------------------------------
nClusers = 50;            % the number of workers
myCluster=parcluster('local');
myCluster.NumWorkers=nClusers;
parpool(myCluster,nClusers)

KK = [5 10 20 30 50]; % sparsity

% -------------------------------------------------------------------------
% Metrics
% -------------------------------------------------------------------------
M = 5;                               % the number of methods for comparision
avg_err = zeros(M,length(KK));       % error
avg_f1 = zeros(M,length(KK));        % f1 score
avg_precision = zeros(M,length(KK)); % precision
avg_recall = zeros(M,length(KK));    % recall

% -------------------------------------------------------------------------
% Estimate the true signal beta
% -------------------------------------------------------------------------
tic
for i = 1:length(KK)
    i
    K = KK(i);         % sparsity level
    B = 100;           % replication number
    p = 5000;          % Signal dimension
    n = 1600;          % local sample size
    N = 25600;         % total sample size
    L = N/n;           % number of machine
    rho = 0.5;         % coorelation
    Maxiter_dist = 10; % maximum number of distributed iterations
    rflipmodel = [1/4; 1/8];
    sigmamodel = [0.1; 0.2];



    % metrics
    error = zeros(M,1);
    f1 = zeros(M,1);
    precision = zeros(M,1);
    recall  = zeros(M,1);
    parfor jj = 1:B
        rng(jj);                                                  % replication
        if(mod(jj,10)==0)
            jj 
        end
        index = int16(rand(L, 1)<0.5)+1;
        rflip = rflipmodel(index);
        sigma = sigmamodel(index);
        [X, y, betaT, supp, Xaug, yaug, tau2] = dataGen(N, L, 'p', p, ...
            'K', K, 'rflip', rflip,  'sigma', sigma, 'rho', rho); % generate data
        [betaE, jjerror, jjf1, jjprecision, jjrecall] = fit(X, y, n, ...
            betaT, Xaug,yaug, K, tau2, 'MAXITER', 5);             % model fitting
        error = error + jjerror;
        f1 = f1 + jjf1;
        precision = precision + jjprecision;
        recall = recall + jjrecall;
    end
    error = error/B;
    f1 = f1/B;
    precision = precision/B;
    recall = recall/B;
    avg_err(:,i) = error;
    avg_f1(:,i) = f1;
    avg_precision(:,i) = precision;
    avg_recall(:,i) = recall;
end % Elapsed time is 3897.633716 seconds.
t = toc;toc
avg_err
avg_f1

delete(gcp('nocreate'))

%% output the latex table
avg_err_f1 = zeros(length(KK),(M - 1)*2 + 1);
avg_err_f1(:,(1:(M - 1))*2+1) = avg_err(2:end,:)';
avg_err_f1(:,(1:(M - 1))*2) = avg_f1(2:end,:)';
avg_err_f1(:,1) = KK(:);
% for latexTable input
input.data = avg_err_f1;
input.dataFormat = {'%d',1,'%.4f',(M - 1)*2}; % format
input.tableBorders = 0;                       % turn off table borders
input.makeCompleteLatexDocument = 0;
latex = latexTable(input);

clear X y;
% save data
save("./output/compare_sparsity-"+char(datetime(now,'ConvertFrom','datenum','Format','yyyy-MM-dd-HH-mm-ss'))+".mat");
dlmwrite("output/compare_sparsity-"+char(datetime(now,'ConvertFrom','datenum','Format','yyyy-MM-dd-HH-mm-ss'))+".txt",char(latex),'delimiter','') % write to file


%% plot
fig1 = figure(1);
plot(KK, avg_err(2,:),'v-.', ...
    KK,avg_err(3,:),'d:', ...
    KK,avg_err(4,:),'>:', ...
    KK,avg_err(5,:),'*-','LineWidth',1)
legend('Average', 'Pooled', 'KSW', 'Distributed','Location','best');
legend off;
xlabel('Sparsity level');
ylabel('$\ell_2$-error',  'Interpreter',  'latex');
title('$\ell_2$-error',  'Interpreter',  'latex');
print(fig1,  '-depsc',  'fig/effect_of_sparsity_error.eps');

fig2 = figure(2);
plot(KK,avg_f1(2,:),'v-.', ...
    KK,avg_f1(3,:),'d:', ...
    KK,avg_f1(4,:),'>:', ...
    KK,avg_f1(5,:),'*-','LineWidth',1)
legend('Average', 'Pooled', 'KSW', 'Distributed','Location','best');
legend off;
xlabel('Sparsity level');
ylabel('$F_1$-score',  'Interpreter',  'latex');
title('$F_1$-score',  'Interpreter',  'latex');
print(fig2,  '-depsc',  'fig/effect_of_sparsity_f1.eps');