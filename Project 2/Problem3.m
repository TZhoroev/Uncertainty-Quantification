clear 
close all
load SIR.txt
t_data = SIR(:,1)';
Infected_data= SIR(:,2)';
Y0=[900;100;0];
N = 1000;
q0 = [.3,.5,.5];
q=lsqnonlin(@(q) odeparameterestimation(t_data,Infected_data,q),q0,[0,0,0],[1,1,1]);


ode_options = odeset('RelTol',1e-8);
t_vals = 0:0.05:5;
params = q;
[~,Y] = ode45(@SIR_rhs,t_vals,Y0,ode_options,params);

figure(1)
plot(t_vals,Y(:,2),'-k','linewidth',5)
hold on
scatter(t_data,Infected_data,'om','MarkerEdgeAlpha',0.5,'linewidth',3)
hold off
set(gca,'Fontsize',20);
xlabel('Time (days)')
ylabel('Number of infection')
legend('Model','Observation','Location','Northeast')


function Error = odeparameterestimation(t_data,Infected_data,params)
Y0 = [900; 100; 0];
N = 1000;
gamma = params(1);  delta = params(2); r = params(3);
ode_options = odeset('RelTol',1e-8);
[~,y] = ode45(@(t,y) [delta*(N-y(1))-gamma*y(2)*y(1); gamma*y(2)*y(1)-(r + delta)*y(2); r*y(2) - delta*y(3)], t_data, Y0,ode_options);
Error = y(:,2)'-Infected_data;
%lse = Error*Error';
end

function dy = SIR_rhs(~,y,params)
N = 1000;
gamma = params(1);  delta = params(2); r = params(3);
S = y(1);          I = y(2);           R = y(3);

dy = [delta*(N-S)-gamma*I*S;
      gamma*I*S-(r + delta)*I;
      r*I - delta*R];
enn