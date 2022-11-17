clear 
close all
load Helmholtz.txt
p_data = Helmholtz(:,1);
psi_data = Helmholtz(:,2);
n = 81;
p = 3;
params_init = [-389.4,761.3, 61.5];
modelfun = @(params) helmholtz(psi_data,p_data,params);
q = fminsearch(modelfun, params_init );

psi  = @(params,p_data) params(1)*p_data.^2 + params(2)*p_data.^4 + params(3)*p_data.^6;

psi_vals = psi(q, p_data);

res = psi_vals - psi_data;

Sens_mat = [p_data.^2 p_data.^4 p_data.^6];

sigma2 = (1/(n-p))*(res'*res);

V = sigma2*eye(p) / (Sens_mat' * Sens_mat); %pxp

data.xdata = p_data;
data.ydata = psi_data; % nx1
tcov = V;

params = {
{'\alpha_1',q(1), [], 0 }
{'\alpha_{11}',q(2), 0 }
{'\alpha_{111}',q(3), 0 }};
model.ssfun = @SS_helmholtz;
model.sigma2 = sigma2;
options.qcov = tcov;
options.nsimu = 10000;
options.updatesigma = 1;
N = 10000;

%%
% Run DRAM to construct the chains (stored in chain) and measurement
% variance (stored in s2chain).
%

[results,chain,s2chain] = mcmcrun(model,data,params,options);



alpha_1_vals = chain(:,1);
alpha_11_vals = chain(:,2);
alpha_111_vals = chain(:,3);


[~,density_alpha_1,alpha_1_mesh,~]=kde(alpha_1_vals);
[~,density_alpha_11,alpha_11_mesh,~]=kde(alpha_11_vals);
[~,density_alpha_111,alpha_111_mesh,~]=kde(alpha_111_vals);
%%
figure(1); clf
mcmcplot(chain,[],results,'chainpanel');

figure(2); clf
mcmcplot(chain,[],results,'pairs');

cov(chain)
chainstats(chain,results)

figure(3); clf
plot(alpha_1_vals,'-','linewidth',2)
set(gca,'Fontsize',22);
axis([0 N  -416 -338])
box on
xlabel('Chain Iteration')
ylabel('Parameter \alpha_1')

figure(4); clf
plot(alpha_11_vals,'-','linewidth',2)
set(gca,'Fontsize',22);
axis([0 N 505 860])
box on
xlabel('Chain Iteration')
ylabel('Parameter \alpha_{11}')

figure(5); clf
plot(alpha_111_vals,'-','linewidth',2)
set(gca,'Fontsize',22);
axis([0 N 0  400])
box on
xlabel('Chain Iteration')
ylabel('Parameter \alpha_{111}')


figure(6); clf
hold on
plot(alpha_1_mesh,density_alpha_1,'k-','linewidth',3)
set(gca,'Fontsize',22);
box on
xlabel('Parameter \alpha_1')
ylabel("PDF")
hold off

figure(7); clf
hold on
plot(alpha_11_mesh,density_alpha_11,'k-','linewidth',3)
set(gca,'Fontsize',22);
xlabel('Parameter \alpha_{11}')
ylabel("PDF")
hold off

figure(8); clf
hold on
plot(alpha_111_mesh,density_alpha_111,'k-','linewidth',3)
set(gca,'Fontsize',22);
box on
xlabel('Parameter \alpha_{111}')
ylabel("PDF")
hold off

figure(9); clf
scatter(alpha_1_vals,alpha_11_vals)
box on
set(gca,'Fontsize',23);
axis([-416 -338 505 860])
box on
xlabel('Parameter \alpha_{1}')
ylabel('Parameter \alpha_{11}')

figure(10); clf
scatter(alpha_1_vals,alpha_111_vals)
box on
set(gca,'Fontsize',23);
axis([-416 -338 0 400])
box on
xlabel('Parameter \alpha_{1}')
ylabel('Parameter \alpha_{111}')

figure(11); clf
scatter(alpha_11_vals,alpha_111_vals)
box on
set(gca,'Fontsize',23);
axis([505 860 0 400])
box on
xlabel('Parameter \alpha_{11}')
ylabel('Parameter \alpha_{111}')

figure(12)
plot(s2chain)
set(gca,'Fontsize',22);
axis([0 N 7 25])
box on
title('Measurement Error Variance \sigma^2')
xlabel('Chain Iteration')
ylabel(' \sigma^2')

mean(s2chain)


%
% Set MCMC parameters
%
N = 1e+5;
R = chol(V);
q_old = q';
SS_old = res'*res;
n0 = 1;
sigma02 = sigma2;
aval = 0.5*(n0 + n);
bval = 0.5*(n0*sigma02 + SS_old);
sigma2 = 1/gamrnd(aval,1/bval);
accept = 0;

%
%  Run the Metropolis algorithm for N iterations.
%

for i = 1:N
z = randn(p,1); 
q_new = q_old + R'*z;
alpha_1 = q_new(1,1);
alpha_11 = q_new(2,1);
alpha_111 = q_new(3,1);
params = [alpha_1 alpha_11 alpha_111];
psi_vals = psi(params,p_data);
res = psi_data - psi_vals;
SS_new = res'*res;
u_alpha = rand(1);
term = exp(-.5*(SS_new-SS_old)/sigma2);
alpha = min(1,term);
if u_alpha < alpha
  Q_MCMC(:,i) = [alpha_1; alpha_11; alpha_111];
  q_old = q_new;
  SS_old = SS_new;
  accept = accept + 1;
else
  Q_MCMC(:,i) = q_old;
end
Sigma2(i) = sigma2;
bval = 0.5*(n0*sigma02 + SS_old);
sigma2 = 1/gamrnd(aval,1/bval);
end


alpha_1_vals1 = Q_MCMC(1,:);
alpha_11_vals1 = Q_MCMC(2,:);
alpha_111_vals1 = Q_MCMC(3,:);


[~,density1_alpha_1,alpha_1_mesh1,~]=kde(alpha_1_vals1);
[~,density1_alpha_11,alpha_11_mesh1,~]=kde(alpha_11_vals1);
[~,density1_alpha_111,alpha_111_mesh1,~]=kde(alpha_111_vals1);


figure(13); clf
plot(alpha_1_vals1,'-','linewidth',2)
set(gca,'Fontsize',22);
axis([0 N -430 -338])
box on
xlabel('Chain Iteration')
ylabel('Parameter \alpha_1')

figure(14); clf
plot(alpha_11_vals1,'-','linewidth',2)
set(gca,'Fontsize',22);
axis([0 N 485 960])
box on
xlabel('Chain Iteration')
ylabel('Parameter \alpha_{11}')

figure(15); clf
hold on
plot(alpha_111_vals1,'-','linewidth',2)
hold off
set(gca,'Fontsize',22);
axis([0 N -100 430])
box on
xlabel('Chain Iteration')
ylabel('Parameter \alpha_{111}')


figure(16); clf
hold on
plot(alpha_1_mesh,density_alpha_1,'k-','linewidth',3)
plot(alpha_1_mesh1,density1_alpha_1,'--r','linewidth',3)
set(gca,'Fontsize',22);
axis([-430 -338   0 0.037])
box on
xlabel('Parameter \alpha_1')
ylabel("PDF")
legend("MCMC","DRAM")
hold off

figure(17); clf
hold on
plot(alpha_11_mesh,density_alpha_11,'k-','linewidth',3)
plot(alpha_11_mesh1,density1_alpha_11,'--r','linewidth',3)
set(gca,'Fontsize',22);
axis([485 960  0 0.0073])
box on
xlabel('Parameter \alpha_{11}')
ylabel("PDF")
legend("MCMC","DRAM")
hold off

figure(18); clf
hold on
plot(alpha_111_mesh,density_alpha_111,'k-','linewidth',3)
plot(alpha_111_mesh1,density1_alpha_111,'--r','linewidth',3)
set(gca,'Fontsize',22);
axis([-100 430 0 .0061])
box on
xlabel('Parameter \alpha_{111}')
ylabel("PDF")
legend("MCMC","DRAM")
hold off

figure(19); clf
scatter(alpha_1_vals1,alpha_11_vals1)
box on
set(gca,'Fontsize',23);
axis([-430 -338 485 960])
box on
xlabel('Parameter \alpha_{1}')
ylabel('Parameter \alpha_{11}')

figure(20); clf
scatter(alpha_1_vals1,alpha_111_vals1)
box on
set(gca,'Fontsize',23);
axis([-430 -338 -100 430])
box on
xlabel('Parameter \alpha_{1}')
ylabel('Parameter \alpha_{111}')

figure(21); clf
scatter(alpha_11_vals1,alpha_111_vals1)
box on
set(gca,'Fontsize',23);
axis([485 960 -100 430])
box on
xlabel('Parameter \alpha_{11}')
ylabel('Parameter \alpha_{111}')

figure(22)
plot(Sigma2)
set(gca,'Fontsize',22);
axis([0 N 7 28])
box on
title('Measurement Error Variance \sigma^2')
xlabel('Chain Iteration')
ylabel(' \sigma^2')

accept/N


function lse = helmholtz(psi_data,p_data,params)
alpha_1 =params(1);
alpha_11 =params(2);
alpha_111 =params(3);
psi_vals = alpha_1*p_data.^2 + alpha_11*p_data.^4 + alpha_111*p_data.^6;
res = psi_vals - psi_data;
lse = res'*res;
end

function lse = SS_helmholtz(params,data)
alpha_1 =params(1);
alpha_11 =params(2);
alpha_111 =params(3);
p_data = data.xdata;
psi_data = data.ydata;
psi_vals = alpha_1*p_data.^2 + alpha_11*p_data.^4 + alpha_111*p_data.^6;
res = psi_vals - psi_data;
lse = res'*res;
end















