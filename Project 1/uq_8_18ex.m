clear 
close all
% params =[theta1, theta2];
Sens_mat1=[ 1 .01];
Sens_mat2=[ 1 .1];
Sens_mat3=[ 1 1];
[vec1, eig1]=eig(Sens_mat1'*Sens_mat1);
[vec2, eig2]=eig(Sens_mat2'*Sens_mat2);
[vec3, eig3]=eig(Sens_mat3'*Sens_mat3);

theta1 = .9+ .2.*randn(1000,1);
theta2 = .9+ .2.*randn(1000,1);

Y = theta1 + theta2;
[Y_density,Y_mesh]= ksdensity(Y);

Y = theta1+1;
[Y_density1,Y_mesh1]= ksdensity(Y);

plot(Y_mesh, Y_density, "-k",Y_mesh1, Y_density1, "--r" )

%%
N=10^3;
E = 90+ 20*rand(N,1);

c = 0.09+.02*rand(N,1);

de_dt =.1;
e=.001;
S_mean_E = zeros(N,1);
S_mean_c = zeros(N,1);
for i=1:N
    S_mean_E(i) = mean(E(i)*e + c*de_dt);
    S_mean_c(i) = mean(E*e + c(i)*de_dt);
end
figure(1)
scatter(E,S_mean_E,'filled')
figure(2)
scatter(c,S_mean_c,'filled')





