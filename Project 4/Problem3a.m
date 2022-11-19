clear 
close all

% given data and parameters
load SIR.txt
t_data = SIR(:,1)';
Infected_data = SIR(:,2)';
Y0 = [900;100;0];
%N = 1000;
gamma = 0.01;
delta = 0.1953;
r = 0.7970;
ode_options = odeset('RelTol',1e-8);
%t_vals = 0:0.05:5;
t_vals = t_data;
params =[gamma delta r];
n = 51;
p = 3;
% 
[~,Y] = ode45(@SIR_rhs,t_data,Y0,ode_options,params);
I = Y(:,2);

res = I' - Infected_data; % 1xn

h = 1e-16;

gamma_complex = complex(gamma,h);
params = [gamma_complex  delta r];
[~,Y] = ode45(@SIR_rhs,t_vals, Y0, ode_options, params);
S_gamma = imag(Y(:,1))/h; I_gamma = imag(Y(:,2))/h; R_gamma = imag(Y(:,3))/h;
  
delta_complex = complex(delta,h);
params = [gamma delta_complex r];
[~,Y] = ode45(@SIR_rhs,t_vals,Y0,ode_options,params);
S_delta = imag(Y(:,1))/h; I_delta = imag(Y(:,2))/h; R_delta = imag(Y(:,3))/h;

r_complex = complex(r,h);
params = [gamma delta r_complex];
[~,Y] = ode45(@SIR_rhs, t_vals,Y0,ode_options,params);
S_r = imag(Y(:,1))/h; I_r = imag(Y(:,2))/h; R_r = imag(Y(:,3))/h;


Sens_mat = [S_gamma S_delta S_r;
            I_gamma I_delta I_r;
            R_gamma R_delta R_r]; % nxp
        
sigma2 = (1/(n-p))*(res*res');

V = sigma2*eye(3)/(Sens_mat' * Sens_mat); %pxp
alpha = 0.005;
t = tinv(1-(alpha/2),n-p);
gamma_s = linspace((gamma - t*sqrt(V(1,1))), (gamma + t*sqrt(V(1,1))), 1000);
dist_gamma = normpdf(gamma_s, gamma,sqrt(V(1,1)));
delta_s = linspace((delta - t*sqrt(V(2,2))), (delta + t*sqrt(V(2,2))), 1000);
dist_delta = normpdf(delta_s, delta,sqrt(V(2,2)));
r_s = linspace((r - t*sqrt(V(3,3))), (r + t*sqrt(V(3,3))), 1000);
dist_r = normpdf(r_s, r,sqrt(V(3,3)));

clear data model options

data.xdata = t_data';
data.ydata = Infected_data'; % nx1
tcov = V;
q_int = [gamma; delta; r];

params = {
{'gamma',q_int(1), 0, 1}
{'delta',q_int(2), 0, 1}
{'r',q_int(3), 0, 1 }};
model.ssfun = @SS_SIR;
model.sigma2 = sigma2;
options.qcov = tcov;
options.nsimu = 10000;
options.updatesigma = 1;
N = 10000;

%%
% Run DRAM to construct the chains (stored in chain) and measurement
% variance (stored in s2chain).
[results,chain,s2chain] = mcmcrun(model,data,params,options);

gamma_vals = chain(:,1);
delta_vals = chain(:,2);
r_vals = chain(:,3);

[~,density_gamma,gamma_mesh,~]=kde(gamma_vals);
[~,density_delta,delta_mesh,~]=kde(delta_vals);
[~,density_r,r_mesh,~]=kde(r_vals);
%%
figure(1); clf
mcmcplot(chain,[],results,'chainpanel');

figure(2); clf
mcmcplot(chain,[],results,'pairs');

cov(chain)
chainstats(chain,results)

figure(3); clf
plot(gamma_vals,'-','linewidth',2)
set(gca,'Fontsize',22);
axis([0 N 0.009 0.011])
box on
xlabel('Chain Iteration')
ylabel('Parameter \gamma')

figure(4); clf
plot(delta_vals,'-','linewidth',2)
set(gca,'Fontsize',22);
axis([0 N 0.16 0.24])
box on
xlabel('Chain Iteration')
ylabel('Parameter \delta')

figure(5); clf
plot(r_vals,'-','linewidth',2)
set(gca,'Fontsize',22);
axis([0 N 0.75 0.85])
box on
xlabel('Chain Iteration')
ylabel('Parameter r')

figure(6); clf
hold on
plot(gamma_mesh,density_gamma,'k-','linewidth',3)
plot(gamma_s,dist_gamma,'--r','linewidth',3)
set(gca,'Fontsize',22);
axis([0.009 0.011  0 3000])
box on
xlabel('Parameter \gamma')
ylabel("PDF")
legend("DRAM","OLS")
hold off

figure(7); clf
hold on
plot(delta_mesh,density_delta,'k-','linewidth',3)
plot(delta_s,dist_delta,'--r','linewidth',3)
set(gca,'Fontsize',22);
axis([0.16 0.24  0 79])
box on
xlabel('Parameter \delta')
ylabel("PDF")
legend("DRAM","OLS")
hold off

figure(8); clf
hold on
plot(r_mesh,density_r,'k-','linewidth',3)
plot(r_s,dist_r,'--r','linewidth',3)
set(gca,'Fontsize',22);
axis([0.75 0.85 0 44])
box on
xlabel('Parameter r')
ylabel("PDF")
legend("DRAM","OLS")
hold off

figure(9); clf
scatter(gamma_vals,delta_vals)
box on
set(gca,'Fontsize',23);
xlabel('Parameter \gamma')
ylabel('Parameter \delta')
axis([0.009 0.011  0.16 0.24])


figure(10); clf
scatter(gamma_vals,r_vals)
box on
set(gca,'Fontsize',23);
axis([0.009 0.011  0.75 0.85])
xlabel('Parameter \gamma')
ylabel('Parameter r')

figure(11); clf
scatter(delta_vals,r_vals)
box on
set(gca,'Fontsize',23);
axis([0.16 0.24  0.75 0.85])
xlabel('Parameter \delta')
ylabel('Parameter r')

figure(12)
plot(s2chain)
hold on
set(gca,'Fontsize',22);
axis([0 N 200 1000])
title('Measurement Error Variance \sigma^2')
xlabel('Chain Iteration')
ylabel(' \sigma^2')

out = mcmcpred(results,chain,s2chain,t_data',@SIR_I,2000); % data must by column maytrix
figure(13)
mcmcpredplot(out);
hold on
set(gca,'Fontsize',24);
plot(t_data, Infected_data, '*r', 'linewidth',3)
hold off

function lse = SS_SIR(params, data)
t_data = data.xdata;
Infected_data = data.ydata;
Y0 = [900; 100; 0];
N = 1000;
gamma = params(1);  delta = params(2); r = params(3);
ode_options = odeset('RelTol',1e-8);
[~,y] = ode45(@(t,y) [delta*(N-y(1))-gamma*y(2)*y(1); gamma*y(2)*y(1)-(r + delta)*y(2); r*y(2) - delta*y(3)], t_data, Y0,ode_options);
Error = y(:,2)-Infected_data;
lse = Error'*Error;
end

function I = SIR_I(data, params)
      t_data = data;
      Y0 = [900; 100; 0];
      N = 1000;
      gamma = params(1);  delta = params(2); r = params(3);
      ode_options = odeset('RelTol',1e-8);
      [~,y] = ode45(@(t,y) [delta*(N-y(1))-gamma*y(2)*y(1); gamma*y(2)*y(1)-(r + delta)*y(2); r*y(2) - delta*y(3)], t_data, Y0,ode_options);
      I = y(:,2);
end

function dy = SIR_rhs(~,y,params)
      N = 1000;
      gamma = params(1);  delta = params(2); r = params(3);
      S = y(1);          I = y(2);           R = y(3);

      dy = [delta*(N-S)-gamma*I*S;
            gamma*I*S-(r + delta)*I;
            r*I - delta*R];
end