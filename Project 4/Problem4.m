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
        
S = [I_gamma I_delta I_r;]; % nxp        
        
sigma2 = (1/(n-p))*(res*res');
V = sigma2*eye(3)/(S' * S); %pxp

var_mod = S*V*S';
V_sigma = sigma2*eye(n);
varY = var_mod + V_sigma;

sd_bd1 = 2*sqrt(diag(varY));
sd_bd2= 2*sqrt(diag(var_mod));

x_plot =[t_data, fliplr(t_data)];
conf_int=[(I-sd_bd2)', flipud(I+sd_bd2)'];
pred_int=[(I-sd_bd1)', flipud(I+sd_bd1)'];
figure(1)
hold on
plot(t_data,Infected_data,'xr','linewidth',2)
fill(x_plot, pred_int, 1,'facecolor', 'black', 'edgecolor', 'none', 'facealpha', 0.4);
fill(x_plot, conf_int, 1,'facecolor', 'blue', 'edgecolor', 'none', 'facealpha', 0.3);
plot(t_data, I,'-k','linewidth',1)
hold off
box on
set(gca,'FontSize',[20])
legend('Data',' 2\sigma_Y Interval','2\sigma_f  Interval','Mean Response','Location','NorthEast')
xlabel('Time')
ylabel('Infection')


function dy = SIR_rhs(~,y,params)
N = 1000;
gamma = params(1);  delta = params(2); r = params(3);
S = y(1);          I = y(2);           R = y(3);

dy = [delta*(N-S)-gamma*I*S;
      gamma*I*S-(r + delta)*I;
      r*I - delta*R];
end