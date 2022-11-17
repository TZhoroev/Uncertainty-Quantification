clear 
close all

tf = 5;
dt = 0.1;
t_vals = 0:dt:tf;
S0 = 900; R0 = 0; I0 = 100;  N=1000;
S_gamma0 = 0; I_gamma0 = 0; R_gamma0 = 0;
S_k0 = 0; I_k0 = 0; R_k0 = 0;
S_delta0 = 0; I_delta0 = 0; R_delta0 = 0;
S_r0 = 0; I_r0 = 0; R_r0 = 0;

gamma = 0.2; k = 0.1; delta = 0.15; r = 0.6;
params=[gamma k delta r];
Y0 = [S0; I0; R0; S_gamma0; I_gamma0; R_gamma0; S_k0; I_k0; R_k0; S_delta0; I_delta0; R_delta0;  S_r0; I_r0; R_r0];
% Integrate the coupled system using ode45.m.  There are now N(p+1) coupled
% differential equations comprised of the state and sensitivity equations
ode_options = odeset('RelTol',1e-8);
[~,Y] = ode45(@SIR_sens,t_vals,Y0,ode_options,params);
% Extract the states and sensitivities. 
S = Y(:,1);             I = Y(:,2);                  R = Y(:,3);
S_gamma_sen = Y(:,4);   I_gamma_sen = Y(:,5);        R_gamma_sen = Y(:,6);
S_k_sen = Y(:,7);       I_k_sen = Y(:,8);            R_k_sen = Y(:,9);
S_delta_sen = Y(:,10);  I_delta_sen = Y(:,11);       R_delta_sen = Y(:,12);
S_r_sen = Y(:,13);      I_r_sen = Y(:,14);           R_r_sen = Y(:,15);
% Compute the sensitivities using complex-step derivative approximations.  This
% requires p=4 solutions of the ODE system. 
clear Y0
h = 1e-16;
Y0 = [S0; I0; R0];

gamma_complex = complex(gamma,h);
params = [gamma_complex k delta r];
[~,Y] = ode45(@SIR_rhs,t_vals, Y0, ode_options, params);
S_gamma = imag(Y(:,1))/h; I_gamma = imag(Y(:,2))/h; R_gamma = imag(Y(:,3))/h;


k_complex = complex(k,h);
params = [gamma k_complex delta r];
[~,Y] = ode45(@SIR_rhs,t_vals,Y0,ode_options,params);
S_k = imag(Y(:,1))/h; I_k = imag(Y(:,2))/h; R_k = imag(Y(:,3))/h;
  
delta_complex = complex(delta,h);
params = [gamma k delta_complex r];
[~,Y] = ode45(@SIR_rhs,t_vals,Y0,ode_options,params);
S_delta = imag(Y(:,1))/h; I_delta = imag(Y(:,2))/h; R_delta = imag(Y(:,3))/h;



r_complex = complex(r,h);
params = [gamma k delta r_complex];
[t,Y] = ode45(@SIR_rhs, t_vals,Y0,ode_options,params);
S_r = imag(Y(:,1))/h; I_r = imag(Y(:,2))/h; R_r = imag(Y(:,3))/h;


figure(4)
plot(t,S_gamma_sen','-b',t,S_gamma,'--r','linewidth',3)
set(gca,'Fontsize',22);
xlabel('Time')
ylabel('S_\gamma')
legend('Sensitivity Eq','Complex-Step','Location','SouthEast')


figure(5)
plot(t,I_gamma_sen,'-b',t,I_gamma,'--r','linewidth',3)
set(gca,'Fontsize',22);
xlabel('Time')
ylabel('I_\gamma')
legend('Sensitivity Eq','Complex-Step','Location','NorthEast')

figure(6)
plot(t,R_gamma_sen,'-b',t,R_gamma,'--r','linewidth',3)
set(gca,'Fontsize',22);
xlabel('Time')
ylabel('R_\gamma')
legend('Sensitivity Eq','Complex-Step','Location','NorthEast')


figure(7)
plot(t,S_k_sen,'-b',t,S_k,'--r','linewidth',3)
set(gca,'Fontsize',22);
xlabel('Time')
ylabel('S_k')
legend('Sensitivity Eq','Complex-Step','Location','SouthEast')
%  print -depsc S_r

figure(8)
plot(t,I_k_sen,'-b',t,I_k,'--r','linewidth',3)
set(gca,'Fontsize',22);
xlabel('Time')
%  ylabel('$\displaystyle\frac{\partial I}{\partial r}$','interpreter','latex')
ylabel('I_k')
legend('Sensitivity Eq','Complex-Step','Location','NorthEast')

figure(9)
plot(t,R_k_sen,'-b',t,R_k,'--r','linewidth',3)
set(gca,'Fontsize',22);
xlabel('Time')
%  ylabel('$\displaystyle\frac{\partial I}{\partial r}$','interpreter','latex')
ylabel('R_k')
legend('Sensitivity Eq','Complex-Step','Location','NorthEast')

figure(10)
plot(t,S_delta_sen,'-b',t,S_delta,'--r','linewidth',3)
set(gca,'Fontsize',22);
xlabel('Time')
ylabel('S_\delta')
legend('Sensitivity Eq','Complex-Step','Location','SouthEast')

figure(11)
plot(t,I_delta_sen','-b',t,I_delta,'--r','linewidth',3)
set(gca,'Fontsize',22);
xlabel('Time')
ylabel('I_\delta')
legend('Sensitivity Eq','Complex-Step','Location','SouthEast')

figure(12)
plot(t,R_delta_sen','-b',t,R_delta,'--r','linewidth',3)
set(gca,'Fontsize',22);
xlabel('Time')
ylabel('R_\delta')
legend('Sensitivity Eq','Complex-Step','Location','NorthEast')

figure(13)
plot(t,S_r_sen,'-b',t,S_r,'--r','linewidth',3)
set(gca,'Fontsize',22);
xlabel('Time')
ylabel('S_r')
legend('Sensitivity Eq','Complex-Step','Location','SouthEast')
%  print -depsc S_r

figure(14)
plot(t,I_r_sen,'-b',t,I_r,'--r','linewidth',3)
set(gca,'Fontsize',22);
xlabel('Time')
%ylabel('$\displaystyle\frac{\partial I}{\partial r}$','interpreter','latex')
ylabel('I_r')
legend('Sensitivity Eq','Complex-Step','Location','NorthEast') 

figure(15)
plot(t,R_r_sen,'-b',t,R_r,'--r','linewidth',3)
set(gca,'Fontsize',22);
xlabel('Time')
%ylabel('$\displaystyle\frac{\partial I}{\partial r}$','interpreter','latex')
ylabel('R_r')
legend('Sensitivity Eq','Complex-Step','Location','NorthEast') 


 
Sens_mat = [R_gamma R_k R_delta R_r];  
eta=1e-10;   
[Id, UnId] = PSS_eig(Sens_mat,eta);
  
  
function dy = SIR_rhs(~,y,params)
N = 1000;
gamma = params(1); k = params(2); delta = params(3); r = params(4);
S = y(1);          I = y(2);           R = y(3);

dy = [delta*(N-S)-gamma*k*I*S;
      gamma*k*I*S-(r + delta)*I;
      r*I - delta*R];
end



function dy = SIR_sens(~, y,params)
N=1000;
gamma = params(1); k = params(2); delta = params(3); r = params(4);
S = y(1);          I = y(2);           R = y(3);
S_gamma = y(4);    I_gamma = y(5);     R_gamma = y(6);
S_k= y(7);         I_k = y(8);         R_k = y(9);
S_delta = y(10);   I_delta = y(11);    R_delta = y(12);
S_r = y(13);       I_r = y(14);        R_r = y(15);

Svec = [S_gamma; I_gamma; R_gamma; S_k; I_k; R_k; S_delta; I_delta; R_delta;  S_r; I_r; R_r];

J = [-(delta+gamma*k*I)  -gamma*k*S  0; gamma*k*I  (gamma*k*S - (r+delta)) 0; 0  r  -delta];

Der = blkdiag(J,J,J,J); Grad = [-k*I*S; k*I*S; 0; -gamma*I*S;  gamma*I*S; 0; (N-S);  -I;  -R; 0; -I; I];
Sen = Der*Svec + Grad;
dy = [delta*(N-S)-gamma*k*I*S;
      gamma*k*I*S-(r + delta)*I;
      r*I - delta*R;
      Sen];
end


function [Id, UnId] = PSS_eig(Sens_mat, eta)

% p number of parameters.
[~,p]=size(Sens_mat);
%Assume all of the parameters are identifiable.
Id=1:p; 
for k=1:p
    [V, D]=eig(Sens_mat'*Sens_mat);
    [d,ind] = sort(abs(diag(D)));
    Vs = V(:,ind);
    if d(1) > eta
        break
    else
       [~,y]=max(abs(Vs(:,1))); %find the position of maximum element in singular vector that is the last column of V matrix.
       Sens_mat(:,y)= [];%remove the column corresponding to above position 
       Id(y)=[]; % Since y'th element is not identifiable we remove it from the identifiable element subset
    end         
end

UnId=1:p; %Define the subset for the unidentifiable parameters.
UnId(Id)=[]; % Remove all parameters that is identifiable from UnId set.
end

