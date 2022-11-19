clear 
close all
alpha_1 = -389.4;
alpha_11 = 761.3;
alpha_111 = 61.5;
sigma = 2.2;

psi = @(P) alpha_1*P.^2 +  alpha_11*P.^4 + alpha_111*P.^6;

n = 81;
P_vals = 0:.8/(n-1):.8;
psi_vals = psi(P_vals);
Y_vals = psi_vals + normrnd(0,sigma, 1,n);
res = Y_vals - psi_vals;
sigma2_1 = res*res'/(n-3);
sigma_hat_1 = sqrt(sigma2_1);

n = 801;
P_vals = 0:.8/(n-1):.8;
psi_vals = psi(P_vals);
Y_vals = psi_vals + normrnd(0,sigma, 1,n);
res = Y_vals - psi_vals;
sigma2_3 = res*res'/(n-3);
sigma_hat_3 = sqrt(sigma2_3);

n = 161;
P_vals = 0:.8/(n-1):.8;
psi_vals = psi(P_vals);
Y_vals = psi_vals + normrnd(0,sigma, 1,n);
res = Y_vals - psi_vals;
sigma2_2 = res*res'/(n-3);
sigma_hat_2 = sqrt(sigma2_2);

P_vals = P_vals';
X = [P_vals.^2 P_vals.^4 P_vals.^6];

theta_hat = (X'*X)\(X'*Y_vals');
cov_est = sigma^2*eye(3)/(X'*X); 

alpha=0.05;
t = tinv(1-(alpha/2),n-p);

figure(1)
hold on
plot(P_vals, res, 'kx', 'linewidth',5)
plot(P_vals, 0*ones(n,1),'-b',P_vals, 2*sigma_hat_2*ones(n,1),'--r',P_vals, -2*sigma_hat_2*ones(n,1),'--r', 'linewidth',3 )
hold off
set(gca,'Fontsize',20);
xlabel('Polarization')
ylabel('Residuals')
legend('Residue','','2\sigma interval','Location','Northeast')


figure(2)
plot(P_vals',psi_vals,'-k','linewidth',5)
hold on
scatter(P_vals',Y_vals,'om','MarkerEdgeAlpha',0.5,'linewidth',3)
hold off
set(gca,'Fontsize',20);
xlabel('Polarization')
ylabel('Helmholtz energy')
legend('Model','Observation','Location','Northeast')