clear 
close all
alpha_1 = -389.4;
alpha_11 = 761.3;
alpha_111 = 61.5;
P_vals = -.8:0.01:0.8;
psi = @(P) alpha_1*P.^2 +  alpha_11*P.^4 + alpha_111*P.^6;
psi_vals = psi(P_vals);

% (a)
figure(1)
plot(P_vals, psi_vals, '-k','linewidth',3)
set(gca,'Fontsize',22);
xlabel('Polarization')
ylabel('Energy')
legend("Helmholtz energy", "Location","North")

% (b)
P_vals = 0:0.05:0.8;

S = [(P_vals.^2)' (P_vals.^4)' (P_vals.^6)'];

F = S'*S;
[V,Ds] = eig(F);

% (c)
y = @(alpha_1, alpha_11, alpha_111) (alpha_1*.8^3)/3 + (alpha_11*.8^5)/5 + (alpha_111*.8^7)/7;  
% parameters of the Morris Screening
Delta = 1/20; r = 50;
a1 = alpha_1*.8; b1 = alpha_1*1.2; a2 = alpha_11*.8; b2 = alpha_11*1.2; a3 = alpha_111*.8; b3 = alpha_111*1.2;
d = zeros(3,r);

for i=1:r
    alpha_1_hat = a1 +(b1-a1)*rand(1,1); alpha_11_hat = a2 +(b2-a2)*rand(1,1); alpha_111_hat = a3 +(b3-a3)*rand(1,1);
    d(1,i) = (y(alpha_1_hat+Delta,alpha_11_hat,alpha_111_hat)-y(alpha_1_hat,alpha_11_hat,alpha_111_hat))/Delta;
    d(2,i) = (y(alpha_1_hat,alpha_11_hat+Delta,alpha_111_hat)-y(alpha_1_hat,alpha_11_hat,alpha_111_hat))/Delta;
    d(3,i) = (y(alpha_1_hat,alpha_11_hat,alpha_111_hat+Delta)-y(alpha_1_hat,alpha_11_hat,alpha_111_hat))/Delta;   
end
mu_star = mean(abs(d),2);
mu = mean(d,2);
sigma = zeros(3,1);

for i=1:3
    sigma(i)=(d(i,:)-mu(i))*(d(i,:)-mu(i))'/(r-1);
end
% (d)


M =1e4;
%p =  sobolset(3,'Skip',1e4,'Leap',1e1);
%D = net(p,2*M);
D = rand(2*M,3);
D(:,1) = D(:,1)*(b1 - a1) + a1;
D(:,2) = D(:,2)*(b2 - a2) + a2;
D(:,3) = D(:,3)*(b3 - a3) + a3;
A = D(1:M,:);
B = D(M+1:2*M,:);
C1 = A; C1(:,1) = B(:,1);
C2 = A; C2(:,2) = B(:,2);
C3 = A; C3(:,3) = B(:,3);
f_A = zeros(M,1); f_B = zeros(M,1); f_C1 = zeros(M,1); f_C2 = zeros(M,1); f_C3 = zeros(M,1); 
for i=1:M
    f_A(i) = y(A(i,1),A(i,2),A(i,3));
    f_B(i) = y(B(i,1),B(i,2),B(i,3));
    f_C1(i) = y(C1(i,1),C1(i,2),C1(i,3));
    f_C2(i) = y(C2(i,1),C2(i,2),C2(i,3));
    f_C3(i) = y(C3(i,1),C3(i,2),C3(i,3));
end
f_D = [f_A; f_B];
denom = mean(f_D.^2) - mean(f_D)^2;

Sobol1 = 1/M*(f_B'*f_C1-f_B'*f_A)/denom;
Sobol2 = 1/M*(f_B'*f_C2-f_B'*f_A)/denom;
Sobol3 = 1/M*(f_B'*f_C3-f_B'*f_A)/denom;

SobolT1 = (mean(f_A.^2)/2+mean(f_C1.^2)/2-f_A'*f_C1/M)/denom;
SobolT2 = (mean(f_A.^2)/2+mean(f_C2.^2)/2-f_A'*f_C2/M)/denom;
SobolT3 = (mean(f_A.^2)/2+mean(f_C3.^2)/2-f_A'*f_C3/M)/denom;

%%%%%%-%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
f_A_fix = zeros(M,1);
for i=1:M
    f_A_fix(i) = y(A(i,1),A(i,2),alpha_111);
end
[~,f_rand,xi_rand] = kde(f_A); 
[~,f_fix,xi_fix] = kde(f_A_fix); 

figure(1)
plot(xi_rand, f_rand, '-k', xi_fix, f_fix, '--r',  'linewidth',3)
set(gca,'Fontsize',22);
xlabel('Energy')
ylabel('Probability density')
legend('All random','\alpha_1, \alpha_{11} random','Location','NorthEast')

    
    



