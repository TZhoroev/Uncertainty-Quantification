clear 
close all
load q_data.txt
f = @(q) (6*q.^2 + 3).* sin(6*q-4);
M = 15;
q_unif = 2*rand(M,1); % a+(b-a)*\xi a=0; b = 2;
f_unif_points = f(q_unif);
q_ls = q_data;
f_ls_points = f(q_ls);
q = 0:.01:2;
f_axis = f(q);

figure(1)
hold on
plot(q,f_axis,'-b','linewidth',3)
plot(q_unif,f_unif_points,'*b',q_ls,f_ls_points,'dr','linewidth',3) 
hold off
set(gca,'Fontsize',22);
xlabel('Parameter q')
ylabel('Model Response')
legend("Response",'Random sample ','Latin hypercube samples','Location','Best')
box on

q_ls = q_unif;
f_ls_points = f(q_ls);
X = [ones(M,1) q_ls q_ls.^2 q_ls.^3 q_ls.^4 q_ls.^5 q_ls.^6 q_ls.^7 q_ls.^8];
u = X\f_ls_points;

fs_regress = u(1)*ones(size(q)) + u(2)*q + u(3)*q.^2 + u(4)*q.^3 + u(5)*q.^4 ...
    + u(6)*q.^5 + u(7)*q.^6 + u(8)*q.^7 + u(9)*q.^8;   

figure(2)
hold on
plot(q,f_axis,'-b', q,fs_regress,'--k','linewidth',3)
plot(q_ls,f_ls_points,'dr','linewidth',3) 
hold off
set(gca,'Fontsize',22);
xlabel('Parameter q')
ylabel('Model Response')
legend("Response"," Surrogate",'Random samples','Location','Best')
box on

q = -.5:.01:2.5;
f_axis = f(q);

fs_regress = u(1)*ones(size(q)) + u(2)*q + u(3)*q.^2 + u(4)*q.^3 + u(5)*q.^4 ...
    + u(6)*q.^5 + u(7)*q.^6 + u(8)*q.^7 + u(9)*q.^8;   

figure(3)
hold on
plot(q,f_axis,'-b', q,fs_regress,'--k','linewidth',3)
plot(q_ls,f_ls_points,'dr','linewidth',3) 
hold off
set(gca,'Fontsize',22);
xlabel('Parameter q')
ylabel('Model Response')
legend("Response"," Surrogate",'Latin hypercube samples','Location','Best')
box on