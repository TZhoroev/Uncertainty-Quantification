clear 
close all
data = [66.04 60.04 54.81 50.42 46.74 43.66 40.76 38.49 36.42 34.77 33.18 32.36 31.56 30.91 30.56];
xdata = [10 14 18 22 26 30 34 38 42 46 50 54 58 62 66];
xvals = 10:.1:70;
u_amb = 22.28;
% Input dimensions and material constants
%
a = 0.95;   % cm
b = 0.95;   % cm
L = 70.0;   % cm
k = 4.01;   % W/cm C
h = 0.0014;
Q = -9.9265; 
n = 15;
p = 2;
% Construct constants and solution
%
gamma = sqrt(2*(a+b)*h/(a*b*k));
gamma_h = (1/(2*h))*gamma;
f1 = exp(gamma*L)*(h + k*gamma);
f2 = exp(-gamma*L)*(h - k*gamma);
f3 = f1/(f2 + f1);
f1_h = exp(gamma*L)*(gamma_h*L*(h+k*gamma) + 1 + k*gamma_h);
f2_h = exp(-gamma*L)*(-gamma_h*L*(h-k*gamma) + 1 - k*gamma_h);
c1 = -Q*f3/(k*gamma);
c2 = Q/(k*gamma) + c1;
f4 = Q/(k*gamma*gamma);
den2 = (f1+f2)^2;
f3_h = (f1_h*(f1+f2) - f1*(f1_h+f2_h))/den2;
c1_h = f4*gamma_h*f3 - (Q/(k*gamma))*f3_h;
c2_h = -f4*gamma_h + c1_h;
c1_Q = -(1/(k*gamma))*f3;
c2_Q = (1/(k*gamma)) + c1_Q;

uvals = c1*exp(-gamma*xvals) + c2*exp(gamma*xvals) + u_amb;
uvals_data = c1*exp(-gamma*xdata) + c2*exp(gamma*xdata) + u_amb;
uvals_Q_data = c1_Q*exp(-gamma*xdata) + c2_Q*exp(gamma*xdata);
uvals_h_data = c1_h*exp(-gamma*xdata) + c2_h*exp(gamma*xdata) + gamma_h*xdata.*(-c1*exp(-gamma*xdata) + c2*exp(gamma*xdata));

res = data - uvals_data;

sens_mat = [uvals_Q_data; uvals_h_data];
sigma2 = (1/(n-p))*(res*res');
V = sigma2*inv(sens_mat*sens_mat');

%
% Set MCMC parameters
%
N = 1e+5;
R = chol(V);
q_old = [Q;h];
SS_old = res*res';
n0 = .1;
sigma02 = sigma2;
aval = 0.5*(n0 + n);
bval = 0.5*(n0*sigma02 + SS_old);
sigma2 = 1/gamrnd(aval,1/bval);
accept = 0;

%
%  Run the Metropolis algorithm for N iterations.
%

for i = 1:N
z = randn(2,1); 
q_new = q_old + R'*z;
Q = q_new(1,1);
h = q_new(2,1);
gamma = sqrt(2*(a+b)*h/(a*b*k));
f1 = exp(gamma*L)*(h + k*gamma);
f2 = exp(-gamma*L)*(h - k*gamma);
f3 = f1/(f2 + f1);
c1 = -Q*f3/(k*gamma);
c2 = Q/(k*gamma) + c1;
uvals_data = c1*exp(-gamma*xdata) + c2*exp(gamma*xdata) + u_amb;
res = data - uvals_data;
SS_new = res*res';
u_alpha = rand(1);
term = exp(-.5*(SS_new-SS_old)/sigma2);
alpha = min(1,term);
if u_alpha < alpha
  Q_MCMC(:,i) = [Q; h];
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

Qvals = Q_MCMC(1,:);
hvals = Q_MCMC(2,:);

%
% Use kde to construct densities for Q and h.
%

range_Q = max(Qvals) - min(Qvals);
range_h = max(hvals) - min(hvals);
Q_min = min(Qvals)-range_Q/10;
Q_max = max(Qvals)+range_Q/10;
h_min = min(hvals)-range_h/10;
h_max = max(hvals)+range_h/10;
[~,density_Q,Qmesh,~]=kde(Qvals);
[~,density_h,hmesh,~]=kde(hvals);
accept/N
figure(1)
plot(Qvals,'-','linewidth',2)
set(gca,'Fontsize',22);
axis([0 N -10.38 -9.52])
box on
xlabel('Chain Iteration')
ylabel('Parameter Q')

figure(2)
plot(hvals,'-','linewidth',2)
set(gca,'Fontsize',22);
axis([0 N 1.36*1e-3 1.49*1e-3])
box on
xlabel('Chain Iteration')
ylabel('Parameter h')

figure(3)
plot(Sigma2)
set(gca,'Fontsize',22);
axis([0 N 0 .6])
box on
title('Measurement Error Variance \sigma^2')
xlabel('Chain Iteration')
ylabel(' \sigma^2')

figure(4)
plot(Qmesh,density_Q,'k-','linewidth',3)
set(gca,'Fontsize',22);
axis([-10.4 -9.5 0 5])
box on
xlabel('Parameter Q')

figure(5)
plot(hmesh,density_h,'k-','linewidth',3)
set(gca,'Fontsize',22);
axis([1.36*1e-3 1.49*1e-3 0 3.1*1e+4])
box on
xlabel('Parameter h')

figure(6)
scatter(Qvals,hvals)
box on
set(gca,'Fontsize',22);
axis([-10.4 -9.5 1.36*1e-3 1.49*1e-3])
box on
xlabel('Parameter Q')
ylabel('Parameter h')


udata = data;
clear data 
a = 0.95;   % cm
b = 0.95;   % cm
L = 70.0;   % cm
k = 4.01;   % W/cm C
h = 0.0014;
Q = -9.9265; 
n = 15;
p = 2;
uvals_data = c1*exp(-gamma*xdata) + c2*exp(gamma*xdata) + u_amb;
uvals_Q_data = c1_Q*exp(-gamma*xdata) + c2_Q*exp(gamma*xdata);
uvals_h_data = c1_h*exp(-gamma*xdata) + c2_h*exp(gamma*xdata) + gamma_h*xdata.*(-c1*exp(-gamma*xdata) + c2*exp(gamma*xdata));
res = udata - uvals_data;

sens_mat = [uvals_Q_data; uvals_h_data];
sigma2 = (1/(n-p))*(res*res');
V = sigma2*inv(sens_mat*sens_mat');

clear data model options

data.xdata = xdata';
data.ydata = udata';
tcov = V;
q_int = [Q; h];

params = {
{'q1',q_int(1), -20}
{'q2',q_int(2), 0}};
model.ssfun = @heatss;
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

%%
% Construct the densities for Q and h.
%

Qvals1 = chain(:,1);
hvals1 = chain(:,2);

[bandwidth_Q1,density_Q1,Qmesh1,~]=kde(Qvals1);
[bandwidth_h1,density_h1,hmesh1,~]=kde(hvals1);

%%
% Plot the results.
%

figure(7); clf
mcmcplot(chain,[],results,'chainpanel');

figure(8); clf
mcmcplot(chain,[],results,'pairs');

cov(chain)
chainstats(chain,results)

figure(9); clf
plot(Qvals1,'-','linewidth',2)
set(gca,'Fontsize',22);
axis([0 N -10.4 -9.5])
box on
xlabel('Chain Iteration')
ylabel('Parameter Q')

figure(10); clf
plot(hvals1,'-','linewidth',2)
set(gca,'Fontsize',22);
axis([0 N 1.36*1e-3 1.49*1e-3])
box on
xlabel('Chain Iteration')
ylabel('Parameter h')

figure(11); clf
hold on
plot(Qmesh,density_Q,'k-','linewidth',3)
plot(Qmesh1,density_Q1,'--r','linewidth',3)
set(gca,'Fontsize',22);
axis([-10.4 -9.5 0 5])
box on
xlabel('Parameter \Phi')
ylabel("PDF")
legend("DRAM","MCMC")
hold off

figure(12); clf
hold on
plot(hmesh,density_h,'k-','linewidth',3)
plot(hmesh1,density_h1,'--r','linewidth',3)
set(gca,'Fontsize',22);
axis([1.36*1e-3 1.49*1e-3 0 3*1e+4])
box on
xlabel('Parameter h')
ylabel("PDF")
legend("DRAM","MCMC")
hold off

figure(13); clf
scatter(Qvals,hvals)
box on
set(gca,'Fontsize',23);
axis([-10.4 -9.5 1.36*1e-3 1.49*1e-3])
xlabel('Parameter Q')
ylabel('Parameter h')

figure(14)
plot(s2chain)
set(gca,'Fontsize',22);
axis([0 N 0 .6])
box on
title('Measurement Error Variance \sigma^2')
xlabel('Chain Iteration')
ylabel('\sigma^2')


function ss = heatss(params,data)

  udata = data.ydata;
  xdata = [10 14 18 22 26 30 34 38 42 46 50 54 58 62 66]';

% Input dimensions and material constants

  a = 0.95;   % cm
  b = 0.95;   % cm
  L = 70.0;   % cm
  k = 4.01;   % W/cm C
  Q = params(1); 
  h = params(2);
  u_amb = 22.28; 

% Construct constants and solution

  gamma = sqrt(2*(a+b)*h/(a*b*k));
  f1 = exp(gamma*L)*(h + k*gamma);
  f2 = exp(-gamma*L)*(h - k*gamma);
  f3 = f1/(f2 + f1);
  c1 = -Q*f3/(k*gamma);
  c2 = Q/(k*gamma) + c1;

  uvals_data = c1*exp(-gamma*xdata) + c2*exp(gamma*xdata) + u_amb;

  res = udata - uvals_data;
  ss = res'*res;
end
