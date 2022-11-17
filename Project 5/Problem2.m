clear 
close all

K = 4;          % Number of tensored Legendre basis functions Psi(q)
N_MC = 1e5;      % Number of Monte Carlo samples
M = 10;          % Number of Gauss-Legendre quadrature points used to approximate discrete projection

mu_k = 8.5;
sigma_k = 0.001;
f = @(k, t) 3*cos(sqrt(k).*t);

nodes = [-0.9739 -0.86506 -0.6794 -0.4334 -0.14887 0.14887 0.4334 0.6794 0.86506 0.9739];
weights = 0.5*[0.06667 0.14945 0.21908 0.2693 0.2955 0.2955 0.2693 0.21908 0.14945 0.06667];
sigmak = 2*sigma_k/sqrt(12);
kvals = mu_k*ones(size(nodes)) + sqrt(3)*sigmak*nodes;
q = zeros(N_MC,1);
q(:,1) = (mu_k - sigma_k) + 2*sigma_k*rand(N_MC,1);

dt = 0.001;                   % d_t = 0.001; Example16_10.m 
t0 = 0;
tf = 5;
t = t0:dt:tf;
N_t = length(t);

poly0 = ones(1,M);                              % P_0(q) = 1
poly1 = nodes;                                  % P_1(q) = q
poly2 = (3/2)*nodes.^2 - (1/2)*ones(1,M);       % P_2(q) = (3/2)q^2 - 1/2
poly3 = (5/2)*nodes.^3 - (3/2)*nodes;           % P_2(q) = (3/2)q^2 - 1/2
h0 = 1;
h1 = 1/3;
h2 = 1/5;
h3 = 1/7;

y_k0 = zeros(1,N_t);
y_k1 = zeros(1,N_t);
y_k2 = zeros(1,N_t);
y_k3 = zeros(1,N_t);
MC_mean = zeros(1,N_t);
MC_sigma = zeros(1,N_t);

%%
%  Compute the  Monte Carlo and discrete projection values at each of values t.
%

for k = 1:N_t
    
    MC =  f(q,t(k));          % Monte Carlo samples
    MC_mean(k) = (1/N_MC)*sum(MC);                                               % MC mean
    tmp = 0;
    for j=1:N_MC
        tmp = tmp + (MC(j) - MC_mean(k))^2;
    end
    MC_sigma(k) = sqrt((1/(N_MC-1))*tmp);
    
    for j = 1:M
        yval = f(kvals(j), t(k));
        y_k0(k) = y_k0(k) + (1/h0)*yval*poly0(j)*weights(j);
        y_k1(k) = y_k1(k) + (1/h1)*yval*poly1(j)*weights(j);
        y_k2(k) = y_k2(k) + (1/h2)*yval*poly2(j)*weights(j);
        y_k3(k) = y_k3(k) + (1/h3)*yval*poly3(j)*weights(j);
    end
end


Y_k = [y_k0; y_k1; y_k2; y_k3];
gamma = [h0 h1 h2 h3];
var = zeros(1,N_t);
for n = 1:N_t
    var_sum = 0;
    for j = 2:K
        var_sum = var_sum + (Y_k(j,n)^2)*gamma(j);
    end
    var(n) = var_sum;
end

DP_mean = Y_k(1,:);           % Discrete projection mean
DP_sd = sqrt(var);            % Discrete projection standard deviation

%%
%  Plot the response mean and standard deviation as a function of the drive frequency omega_F.
%

figure(1)
plot(t,DP_mean,'k-',t,MC_mean,'--b','linewidth',3)
set(gca,'Fontsize',22);
xlabel('t-vals')
ylabel('Response Mean')
legend('Discrete Projection','Monte Carlo','Location','Best')


figure(2)
plot(t,DP_sd,'-k',t, MC_sigma,'--b','linewidth',3)
set(gca,'Fontsize',22);
xlabel('t-vals')
ylabel('Response Standard Deviation')
legend('Discrete Projection','Monte Carlo','Location','Best')

figure(3)
plot(t,f(mu_k,t),'-k','linewidth',3)
set(gca,'Fontsize',22);
xlabel('t-vals')
ylabel('Response')
legend('Given Function','Location','Best')

