clear 
close all
load q_data.txt
% Create training points and responses along with the test points and their responses

xs = q_data; % training points
Y = @(q) ((6*q.^2 + 3).* sin(6*q-4))';
ys = Y(xs);
%xp = linspace(0,2,200)';
xp = linspace(-.5,2.5,200)';
yp = Y(xp);

% Input initial values for hyperparameters 
%   Book:   ell_k,  sigma,  sigma_0
%   MATLAB: sigmaL, sigmaF, sigma 
%
kparams = [3.5, 10];          % [sigmaL, sigmaF]
% Estimate the hyperparameters using an initial estimate of sigma = eps and compute the expected
% prediction at xp along with the variance
%

gprMdl = fitrgp(xs,ys,'KernelFunction','squaredexponential','KernelParameters',kparams,'Sigma',eps);
[pred,~,yint] = predict(gprMdl,xp);
%
%  Extract the optimized values of sigma, sigmaL, and sigmaF
%  Extract the covariance function for plotting
%
sigmaL = gprMdl.KernelInformation.KernelParameters(1); 
sigmaF = gprMdl.KernelInformation.KernelParameters(2); 
sigma  = gprMdl.Sigma; 
beta = gprMdl.Beta;
kfcn = gprMdl.Impl.Kernel.makeKernelAsFunctionOfXNXM(gprMdl.Impl.ThetaHat);
K = kfcn(xp(1),xp(1:end));

%
% Plot the covariance function and predictions and standard deviation intervals.
%

figure(1)
plot(xp,K,'b-','linewidth',3)
set(gca,'Fontsize',22);
xlabel('Parameter q') 
ylabel('Covariance Function c')


figure(2)
f = [yint(:,2); flipud(yint(:,1))];
h(1) = fill([xp; flipud(xp)], f, [7 7 7]/8);
set(get(get(h(1),'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
hold on
h(2) = plot(xs,ys,'ro','linewidth',5,'DisplayName','Data');
h(3) = plot(xp,pred,'b-','linewidth',3,'DisplayName','Predictive Mean');
hold off
legend('Data','Prediction Mean', '95% Prediction Interval', 'Location','best')
set(gca,'Fontsize',22);
xlabel('Parameter q')
ylabel('Response')