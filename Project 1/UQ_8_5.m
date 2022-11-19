clear 
close all
T = 5;  %Final time
K = 20.5; constant
C = 1.5;
t_vals = 0:.05:T; % equal spaced points
z_0 = 2;
z_dot_0 = -C;
ode_options = odeset('RelTol',1e-8);

y = @(K,C) 2*exp(-C*t_vals'/2).*cos(sqrt(K-C^2/4)*t_vals');
h = 1e-8; % step for finite difference method
% Find dz/dK using Finite difference difference Approximation using
% analytic solution
z_K_finite = (y(K+h,C)-y(K,C))/h;

% Find dz/dC using Finite difference difference Approximation using
% analytic solution
h = 1e-7;
z_C_finite = (y(K,C+h)-y(K,C))/h;

% Using Complex step Approximation
z_K_complex = imag(y(complex(K,h),C))/h;
z_C_complex = imag(y(K,complex(C,h)))/h;


% Find dz/dK using Complex step Approximation
h = 1e-16;
K_complex = complex(K,h); 
params = [K_complex, C];
[~,Y] = ode45(@spring_rhs,t_vals,[z_0, z_dot_0],ode_options,params);
z_K = imag(Y(:,1))/h;

% Find dz/dC using Complex step Approximation
h = 1e-16;
C_complex = complex(C,h); 
params = [K, C_complex];
[t,Y] = ode45(@spring_rhs,t_vals,[z_0, z_dot_0],ode_options,params);
z_C = imag(Y(:,1))/h;


% Analytic Solutions
y_K = exp(-C*t_vals'/2).*(-2*t_vals'/sqrt(4*K-C^2)).*sin(sqrt(K-C^2/4)*t_vals');
y_C = exp(-C*t_vals'/2).*((C*t_vals'/sqrt(4*K-C^2)).*sin(sqrt(K-C^2/4)*t_vals')-t_vals'.*cos(sqrt(K-C^2/4)*t_vals'));

% Verification that all methods produced same solution.
figure(1)
plot(t, z_K_complex, '-k', t, y_K,'--c', t, z_K_finite, 'r*', 'linewidth',3)
set(gca,'Fontsize',22);
xlabel('Time')
ylabel('z_K')
legend('Complex Step','Analytic','Finite Difference','Location','NorthEast') 

figure(2)
plot(t, z_C_complex, '-k', t, y_C,'--c', t, z_C_finite, 'r*', 'linewidth',3)
set(gca,'Fontsize',22);
xlabel('Time')
ylabel('z_C')
legend('Complex Step','Analytic','Finite Difference','Location','NorthEast') 

% Function to solve the ODE system.
function dy = spring_rhs(~, y,params)
K=params(1);
C=params(2);
z=y(1);
z_dot=y(2);
dy=[z_dot;  -C*z_dot-K*z];
end