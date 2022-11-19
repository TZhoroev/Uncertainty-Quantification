clear
close all
x_data = (58:72)'; % given heights
n = length(x_data); % number of the data points
p = 3; % we have three parameters
Y_data = [115  117 120 123 126 129 132 135 139 142 146  150  154 159 164]'; % given weights
theta = [261.88 -88.18 11.96]'; % given parameter value

X = [ones(n,1) x_data/12 (x_data/12).^2]; % design matrix

sigma2 = 0.1491; % used result from  Example 11.7
sigma = sqrt(sigma2);

V_sigma = sigma2* eye(n);
V = (sigma2*eye(p))/(X'*X); % sigma^2*inv(X'*X)

mean = X*theta;
covY = X*V*X' + V_sigma;
sd_bd = 2*sqrt(diag(covY));

x_test = (58:.5:72)';
Y_test = theta(1)+ theta(2)*(x_test/12) + theta(3)*(x_test/12).^2 ;
N = length(x_test);
X_test = [ones(N,1) x_test/12  x_test.^2/144];

alpha = 0.05;
tcrit2 = tinv(1-alpha/2,n-p);
int_upper = zeros(N,1); int_lower = zeros(N,1);
for j=1:N
    int_upper(j) = Y_test(j) + tcrit2*sigma*sqrt(1 + X_test(j,:)*inv(X'*X)*(X_test(j,:)'));
    int_lower(j) = Y_test(j) - tcrit2*sigma*sqrt(1 + X_test(j,:)*inv(X'*X)*(X_test(j,:)'));
end

x_plot_data =[x_data', fliplr(x_data')];
x_plot_test =[x_test', fliplr(x_test')];
conf_int=[(mean-sd_bd)', flipud(mean+sd_bd)'];
pred_int=[(int_lower)', flipud(int_upper)'];

figure(1)
plot(x_test,Y_test,'-k','linewidth',1)
hold on
plot(x_data,Y_data,'*m','linewidth',2)
fill(x_plot_data, conf_int, 1,'facecolor', 'blue', 'edgecolor', 'none', 'facealpha', 0.4);
fill(x_plot_test, pred_int, 1,'facecolor', 'black', 'edgecolor', 'red', 'facealpha', 0.3);
hold off
set(gca,'FontSize',26)
legend('Mean Response','Data','2\sigma Interval','Prediction Interval','Location','NorthEast')
xlabel('Height (in)')
ylabel('Weight (lbs)')

x_test = (50:.5:80)';
Y_test = theta(1)+ theta(2)*(x_test/12) + theta(3)*(x_test/12).^2 ;
N = length(x_test);
X_test = [ones(N,1) x_test/12  x_test.^2/144];

alpha = 0.05;
tcrit2 = tinv(1-alpha/2,n-p);
int_upper = zeros(N,1); int_lower = zeros(N,1);
for j=1:N
    int_upper(j) = Y_test(j) + tcrit2*sigma*sqrt(1 + X_test(j,:)*inv(X'*X)*(X_test(j,:)'));
    int_lower(j) = Y_test(j) - tcrit2*sigma*sqrt(1 + X_test(j,:)*inv(X'*X)*(X_test(j,:)'));
end

x_plot_data =[x_data', fliplr(x_data')];
x_plot_test =[x_test', fliplr(x_test')];
conf_int=[(mean-sd_bd)', flipud(mean+sd_bd)'];
pred_int=[(int_lower)', flipud(int_upper)'];

figure(2)
plot(x_test,Y_test,'-k','linewidth',1)
hold on
plot(x_data,Y_data,'*m','linewidth',2)
fill(x_plot_data, conf_int, 1,'facecolor', 'blue', 'edgecolor', 'none', 'facealpha', 0.4);
fill(x_plot_test, pred_int, 1,'facecolor', 'black', 'edgecolor', 'red', 'facealpha', 0.3);
hold off
set(gca,'FontSize',26)
legend('Mean Response','Data','2\sigma Interval','Prediction Interval','Location','NorthEast')
xlabel('Height (in)')
ylabel('Weight (lbs)')