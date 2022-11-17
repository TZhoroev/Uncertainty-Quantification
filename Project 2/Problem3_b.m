clear 
close all
load SIR.txt
t_data = SIR(:,1)';
t_vals = t_data;
Infected_data= SIR(:,2)';
Y0=[900;100;0];


S0 = 900; R0 = 0; I0 = 100;  N=1000;
gamma = 9.9929e-03;
delta = 1.9529e-01;
r = 7.9698e-01;

ode_options = odeset('RelTol',1e-8);
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
[t,Y] = ode45(@SIR_rhs, t_vals,Y0,ode_options,params);
S_r = imag(Y(:,1))/h; I_r = imag(Y(:,2))/h; R_r = imag(Y(:,3))/h;


Sens_mat = [S_gamma S_delta S_r;
            I_gamma I_delta I_r;
            R_gamma R_delta R_r;];
 
params = [gamma delta r];       
[~,Y] = ode45(@SIR_rhs,t_vals,Y0,ode_options,params);
S = Y(:,1); I = Y(:,2); R = Y(:,3);  

figure(1)
hold on
plot(t_vals,S,'-k','linewidth',4)
plot(t_vals,I,'-.b','linewidth',4)
plot(t_data,R,'--r','linewidth',4)

scatter(t_data,Infected_data,'om','MarkerEdgeAlpha',0.5,'linewidth',3)
hold off
set(gca,'Fontsize',20);
xlabel('Time (days)')
ylabel('Number of infection')
legend('Susceptible','Infected','Recovered','Observation','Location','east')



Residue = Infected_data - I';
p=3;
n=length(Residue);
sigma2 = (Residue*Residue')/(n-p);

sigma = sqrt(sigma2);

figure(3)
hold on
plot(t_vals, Residue, 'kx', 'linewidth',5)
plot(t_vals, 0*ones(n,1),'-b',t_vals, 2*sigma*ones(n,1),'--r',t_vals, -2*sigma*ones(n,1),'--r', 'linewidth',3 )
hold off
set(gca,'Fontsize',20);
xlabel('Time')
ylabel('Residuals')
legend('Residue','','2\sigma interval','Location','Northeast')

cov_est = sigma2*eye(3)/(Sens_mat'*Sens_mat); 
rank(cov_est)

alpha=0.001;
t = tinv(1-(alpha/2),n-p);
gamma_s = linspace((gamma - t*sqrt(cov_est(1,1))), (gamma + t*sqrt(cov_est(1,1))), 500);

dist_gamma = normpdf(gamma_s, gamma,sqrt(cov_est(1,1)));
figure(4)
plot(gamma_s,dist_gamma,'-k', 'linewidth',4)
set(gca,'Fontsize',20);
xlabel('\gamma')
ylabel('PDF')


delta_s = linspace((delta - t*sqrt(cov_est(2,2))), (delta + t*sqrt(cov_est(2,2))), 500);

dist_delta = normpdf(delta_s, delta,sqrt(cov_est(2,2)));
figure(5)
plot(delta_s,dist_delta,'-k', 'linewidth',4)
set(gca,'Fontsize',20);
xlabel('\delta')
ylabel('PDF')

r_s = linspace((r - t*sqrt(cov_est(3,3))), (r + t*sqrt(cov_est(3,3))), 500);

dist_r = normpdf(r_s, r,sqrt(cov_est(3,3)));
figure(6)
plot(r_s,dist_r,'-k', 'linewidth',4)
set(gca,'Fontsize',20);
xlabel('r')
ylabel('PDF')
















function dy = SIR_rhs(~,y,params)
N = 1000;
gamma = params(1);  delta = params(2); r = params(3);
S = y(1);          I = y(2);           R = y(3);

dy = [delta*(N-S)-gamma*I*S;
      gamma*I*S-(r + delta)*I;
      r*I - delta*R];
end