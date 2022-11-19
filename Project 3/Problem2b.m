clear 
close all

% given data and parameters
load SIR.txt
t_data = SIR(:,1)';
Infected_data = SIR(:,2)';
Y0 = [900;100;0];
gamma = 0.029146 ;
delta = 0.19694;
r = 0.79965 ;
k = 0.41883;
ode_options = odeset('RelTol',1e-8);
%t_vals = 0:0.05:5;
t_vals = t_data;
params =[gamma delta r k];
n = 51;
p = 4;

% 
[~,Y] = ode45(@SIR_rhs,t_data,Y0,ode_options,params);
I = Y(:,2);

res = I' - Infected_data; % 1xn

h = 1e-16;

gamma_complex = complex(gamma,h);
params = [gamma_complex  delta r  k];
[~,Y] = ode45(@SIR_rhs,t_vals, Y0, ode_options, params);
S_gamma = imag(Y(:,1))/h; I_gamma = imag(Y(:,2))/h; R_gamma = imag(Y(:,3))/h;

delta_complex = complex(delta,h);
params = [gamma delta_complex r k];
[~,Y] = ode45(@SIR_rhs,t_vals,Y0,ode_options,params);
S_delta = imag(Y(:,1))/h; I_delta = imag(Y(:,2))/h; R_delta = imag(Y(:,3))/h;

r_complex = complex(r,h);
params = [gamma delta r_complex k];
[~,Y] = ode45(@SIR_rhs, t_vals,Y0,ode_options,params);
S_r = imag(Y(:,1))/h; I_r = imag(Y(:,2))/h; R_r = imag(Y(:,3))/h;

k_complex = complex(k,h);
params = [gamma delta r k_complex];
[~,Y] = ode45(@SIR_rhs, t_vals,Y0,ode_options,params);
S_k = imag(Y(:,1))/h; I_k = imag(Y(:,2))/h; R_k = imag(Y(:,3))/h;


Sens_mat = [S_gamma S_delta S_r S_k;
            I_gamma I_delta I_r I_k;
            R_gamma R_delta R_r R_k]; % nxp
        
sigma2 = (1/(n-p))*(res*res');

V = sigma2*eye(4) / diag(diag(Sens_mat' * Sens_mat)); %pxp

clear data model options

data.xdata = t_data';
data.ydata = Infected_data'; % nx1
q_int = [gamma; delta; r; k];

params = {
{'gamma',q_int(1), 0, 1}
{'delta',q_int(2), 0, 1}
{'r',q_int(3), 0, 1 }
{'k',q_int(4),0 ,1 }};
model.ssfun = @SS_SIR;
model.sigma2 = sigma2;
options.nsimu = 10000;
options.updatesigma = 1;
N = 10000;

%%
% Run DRAM to construct the chains (stored in chain) and measurement
% variance (stored in s2chain).
%

[results,chain,s2chain] = mcmcrun(model,data,params,options);


gamma_vals = chain(:,1);
delta_vals = chain(:,2);
r_vals = chain(:,3);
k_vals = chain(:,4);

[~,density_gamma,gamma_mesh,~]=kde(gamma_vals);
[~,density_delta,delta_mesh,~]=kde(delta_vals);
[~,density_r,r_mesh,~]=kde(r_vals);
[~,density_k,k_mesh,~]=kde(k_vals);
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
xlabel('Chain Iteration')
ylabel('Parameter \gamma')

figure(4); clf
plot(delta_vals,'-','linewidth',2)
set(gca,'Fontsize',22);
xlabel('Chain Iteration')
ylabel('Parameter \delta')

figure(5); clf
plot(r_vals,'-','linewidth',2)
set(gca,'Fontsize',22);
xlabel('Chain Iteration')
ylabel('Parameter r')

figure(6); clf
plot(k_vals,'-','linewidth',2)
set(gca,'Fontsize',22);
xlabel('Chain Iteration')
ylabel('Parameter k')


figure(7); clf
hold on
plot(gamma_mesh,density_gamma,'k-','linewidth',3)
set(gca,'Fontsize',22);
xlabel('Parameter \gamma')
ylabel("PDF")
hold off

figure(8); clf
hold on
plot(delta_mesh,density_delta,'k-','linewidth',3)
set(gca,'Fontsize',22);
xlabel('Parameter \delta')
ylabel("PDF")
hold off

figure(9); clf
hold on
plot(r_mesh,density_r,'k-','linewidth',3)
set(gca,'Fontsize',22);
xlabel('Parameter r')
ylabel("PDF")
hold off

figure(10); clf
hold on
plot(k_mesh,density_k,'k-','linewidth',3)
set(gca,'Fontsize',22);
xlabel('Parameter k')
ylabel("PDF")
hold off

figure(11); clf
scatter(gamma_vals,delta_vals)
box on
set(gca,'Fontsize',23);
xlabel('Parameter \gamma')
ylabel('Parameter \delta')

figure(12); clf
scatter(gamma_vals,r_vals)
box on
set(gca,'Fontsize',23);
xlabel('Parameter \gamma')
ylabel('Parameter r')

figure(13); clf
scatter(gamma_vals,k_vals)
box on
set(gca,'Fontsize',23);
xlabel('Parameter \gamma')
ylabel('Parameter k')

figure(14); clf
scatter(delta_vals,r_vals)
box on
set(gca,'Fontsize',23);
xlabel('Parameter \delta')
ylabel('Parameter r')

figure(15); clf
scatter(delta_vals,k_vals)
box on
set(gca,'Fontsize',23);
xlabel('Parameter \delta')
ylabel('Parameter k')

figure(16); clf
scatter(r_vals,k_vals)
box on
set(gca,'Fontsize',23);
xlabel('Parameter r')
ylabel('Parameter k')

figure(17)
plot(s2chain)
set(gca,'Fontsize',22);
title('Measurement Error Variance \sigma^2')

mean(s2chain)


function lse = SS_SIR(params, data)
t_data = data.xdata;
Infected_data = data.ydata;
Y0 = [900; 100; 0];
N = 1000;
gamma = params(1);  delta = params(2); r = params(3); k = params(4);
ode_options = odeset('RelTol',1e-8);
[~,y] = ode45(@(t,y) [delta*(N-y(1))-k*gamma*y(2)*y(1); k*gamma*y(2)*y(1)-(r + delta)*y(2); r*y(2) - delta*y(3)], t_data, Y0,ode_options);
Error = y(:,2)-Infected_data;
lse = Error'*Error;
end


function dy = SIR_rhs(~,y,params)
N = 1000;
gamma = params(1);  delta = params(2); r = params(3); k = params(4);
S = y(1);          I = y(2);           R = y(3);

dy = [delta*(N-S)-k*gamma*I*S;
      k*gamma*I*S-(r + delta)*I;
      r*I - delta*R];
end