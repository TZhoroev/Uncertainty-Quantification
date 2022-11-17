clear 
close all

% given data and parameters
t_data = [0 1 2 3 4 5 6 7 8 9 10 11 12 13];
Infected_data = [3 8 26 76 225 298 258 233 189 128 68 29 14 4];
Y0=[730;3;0];
gamma = 0.0022305;
r =  0.44807;

ode_options = odeset('RelTol',1e-8);
%t_vals = 0:0.05:5;
t_vals = t_data;
params =[gamma r];
n = 14;
p = 2;

% 
[~,Y] = ode45(@SIR_rhs,t_data,Y0,ode_options,params);
I = Y(:,2);

res = I' - Infected_data; % 1xn

h = 1e-16;

gamma_complex = complex(gamma,h);
params = [gamma_complex r];
[~,Y] = ode45(@SIR_rhs,t_vals, Y0, ode_options, params);
S_gamma = imag(Y(:,1))/h; I_gamma = imag(Y(:,2))/h; R_gamma = imag(Y(:,3))/h;


r_complex = complex(r,h);
params = [gamma r_complex];
[~,Y] = ode45(@SIR_rhs, t_vals,Y0,ode_options,params);
S_r = imag(Y(:,1))/h; I_r = imag(Y(:,2))/h; R_r = imag(Y(:,3))/h;




Sens_mat = [S_gamma S_r ;
            I_gamma I_r ;
            R_gamma R_r]; % nxp
        
sigma2 = (1/(n-p))*(res*res');

V = sigma2*eye(p) / (Sens_mat' * Sens_mat); %pxp




clear data model options

data.xdata = t_data';
data.ydata = Infected_data'; % nx1
tcov = V;
q_int = [gamma;  r];

params = {
{'gamma',q_int(1), 0, 1}
{'r',q_int(2), 0, 1 }};
model.ssfun = @SS_SIR;
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



gamma_vals = chain(:,1);
r_vals = chain(:,2);


[~,density_gamma,gamma_mesh,~]=kde(gamma_vals);
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
axis([0 N 0.00206 0.0024])
box on
xlabel('Chain Iteration')
ylabel('Parameter \gamma')


figure(4); clf
plot(r_vals,'-','linewidth',2)
set(gca,'Fontsize',22);
axis([0 N 0.38 0.52])
box on
xlabel('Chain Iteration')
ylabel('Parameter r')



figure(5); clf
hold on
plot(gamma_mesh,density_gamma,'k-','linewidth',3)
set(gca,'Fontsize',22);
axis([0.00206 0.0024  0 10010])
box on
xlabel('Parameter \gamma')
ylabel("PDF")
hold off


figure(6); clf
hold on
plot(r_mesh,density_r,'k-','linewidth',3)
set(gca,'Fontsize',22);
axis([0.38 0.52 0 27])
box on
xlabel('Parameter r')
ylabel("PDF")
hold off


figure(7); clf
scatter(gamma_vals,r_vals)
axis([0.00206 0.0024 0.38 0.52])
box on
set(gca,'Fontsize',23);
xlabel('Parameter \gamma')
ylabel('Parameter r')


figure(8)
plot(s2chain)
set(gca,'Fontsize',22);
axis([0 N 100 1800])
box on
title('Measurement Error Variance \sigma^2')
xlabel('Chain Iteration')
ylabel(' \sigma^2')

out = mcmcpred(results,chain,s2chain,t_data',@SIR_I,2000); % data must by column maytrix
figure(9)
mcmcpredplot(out);
hold on
set(gca,'Fontsize',24);
plot(t_data, Infected_data, '*r', 'linewidth',3)
hold off







function lse = SS_SIR(params, data)
t_data = data.xdata;
Infected_data = data.ydata;
Y0=[760;3;0];
N = 763;
gamma = params(1);  delta = 0; r = params(2); 
ode_options = odeset('RelTol',1e-8);
[~,y] = ode45(@(t,y) [delta*(N-y(1))-gamma*y(2)*y(1); gamma*y(2)*y(1)-(r + delta)*y(2); r*y(2) - delta*y(3)], t_data, Y0,ode_options);
Error = y(:,2)-Infected_data;
lse = Error'*Error;
end

function I = SIR_I(data, params)
t_data = data;
Y0=[760;3;0];
N = 763;
gamma = params(1);  delta = 0; r = params(2); 
ode_options = odeset('RelTol',1e-8);
[~,y] = ode45(@(t,y) [delta*(N-y(1))-gamma*y(2)*y(1); gamma*y(2)*y(1)-(r + delta)*y(2); r*y(2) - delta*y(3)], t_data, Y0,ode_options);
I = y(:,2);
end

function dy = SIR_rhs(~,y,params)
N = 763;
gamma = params(1);  delta = 0; r = params(2);
S = y(1);          I = y(2);           R = y(3);

dy = [delta*(N-S)-gamma*I*S;
      gamma*I*S-(r + delta)*I;
      r*I - delta*R];
end