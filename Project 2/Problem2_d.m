clear 
close all
t_data = [0 1 2 3 4 5 6 7 8 9 10 11 12 13];
Infected_data = [3 8 26 76 225 298 258 233 189 128 68 29 14 4];
Y0=[760;3;0];
N=763;

q0 = [1,3];


ode_options = odeset('RelTol',1e-12);
t_vals = 0:0.1:13;%
params = [0.002227742059861, 0.446928402444749];
[~,Y] = ode45(@SIR_rhs,t_vals,Y0,ode_options,params);
S = Y(:,1); I = Y(:,2); R = Y(:,3); 

figure(1)
hold on
plot(t_vals,S,'-k','linewidth',4)
plot(t_vals,I,'-.b','linewidth',4)
plot(t_vals,R,'--r','linewidth',4)
scatter(t_data,Infected_data,'om','MarkerEdgeAlpha',0.5,'linewidth',3)
xlim([0 13])
box on
hold off
set(gca,'Fontsize',20);
xlabel('Time (days)')
ylabel('Number of infection')
legend('Susceptible','Infected','Recovered','Observation','Location','east')

%------------------------------%

gamma= 0.002218323261879;

r= 0.446928402444749;
ode_options = odeset('RelTol',1e-8);
h = 1e-16;

gamma_complex = complex(gamma,h);
params = [gamma_complex r];
[~,Y] = ode45(@SIR_rhs,t_data, Y0, ode_options, params);
S_gamma = imag(Y(:,1))/h; I_gamma = imag(Y(:,2))/h; R_gamma = imag(Y(:,3))/h;

r_complex = complex(r,h);
params = [gamma r_complex];
[~,Y] = ode45(@SIR_rhs, t_data,Y0,ode_options,params);
S_r = imag(Y(:,1))/h; I_r = imag(Y(:,2))/h; R_r = imag(Y(:,3))/h;

Sens_mat = [S_gamma  S_r;
            I_gamma  I_r;
            R_gamma  R_r;];
params = [gamma r];       
[~,Y] = ode45(@SIR_rhs,t_data,Y0,ode_options,params);
S = Y(:,1); I = Y(:,2); R = Y(:,3);    

Residue = Infected_data - I';
p=2;
n=length(Residue);
sigma2 = (Residue*Residue')/(n-p);

sigma = sqrt(sigma2);

figure(3)
hold on
plot(t_data, Residue, 'kx', 'linewidth',5)
plot(t_data, 0*ones(n,1),'-b',t_data, 2*sigma*ones(n,1),'--r',t_data, -2*sigma*ones(n,1),'--r', 'linewidth',3 )
hold off
xlim([0 13])
box on
set(gca,'Fontsize',20);
xlabel('Time')
ylabel('Residuals')
legend('Residue','','2\sigma interval','Location','Northeast')

cov_est = sigma2*eye(2)/(Sens_mat'*Sens_mat); 
rank(cov_est)







function Error = odeparameterestimation(t_data,Infected_data,q)
Y0=[760;3;0];
gamma = q(1);   r = q(2);
[~,Y] = ode45(@(t,y) [-gamma*y(2)*y(1); gamma*y(2)*y(1)-r*y(2); r*y(2)], t_data, Y0);
Error = Y(:,2)'-Infected_data;
%lse  =  Error*Error';
end

function dy = SIR_rhs(~,y,params)
gamma = params(1);   r = params(2);
S = y(1);          I = y(2);         

dy = [-gamma*I*S;
      gamma*I*S - r*I;
      r*I] ;
end