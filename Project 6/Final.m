clear
close all
load db_data.txt
Re_data = db_data(:,1);
Pr_data = db_data(:,2);
Nu_data = db_data(:,3);
theta_1 =  0.023 ;
theta_2 = 0.8;
theta_3 = 0.4;
p = 3;  n = 56;
params = [theta_1  theta_2  theta_3];
Nu_data_vals = Dittus_Boelter([Re_data Pr_data],params);
Res = Nu_data_vals - Nu_data;
sigma2 = (Res'*Res)/(n-p);
sigma_hat = sqrt(sigma2);
figure(14)
hold on
plot(Nu_data_vals, Res, 'kx', 'linewidth',5)
plot(Nu_data_vals, 0*ones(n,1),'-b',Nu_data_vals, 2*sigma_hat*ones(n,1),'-r',Nu_data_vals, -2*sigma_hat*ones(n,1),'-r', 'linewidth',3 )
hold off
set(gca,'Fontsize',20);
xlabel('Nu values')
ylabel('Residuals')
legend('Residue','','2\sigma interval','Location','Northeast')

%%

data.Re_data = Re_data;
data.Pr_data = Pr_data;
data.Nu_data = Nu_data;
options.Algorithm = 'levenberg-marquardt';
q = params;
q=lsqnonlin(@(q) odeparameterestimation(data,q),params,[],[], options);

params=q;

X = [(Re_data.^params(2)).*(Pr_data.^params(3))    params(1)*(Re_data.^params(2)).*(Pr_data.^params(3)).*log(Re_data)   params(1).*(Re_data.^params(2)).*(Pr_data.^params(3)).*log(Pr_data)];
Fisher = X'*X;
eta = 1e-8;
[Id, UnId] = PSS_SVD(X,eta);
[Ida, UnIda] = PSS_SVD2update(X,eta);
% 
% 
% Nu_data_vals = Dittus_Boelter([Re_data Pr_data],params);
% 
% 
% Res = Nu_data_vals - Nu_data;
% sigma2 = (Res'*Res)/(n-p);
% sigma_hat = sqrt(sigma2);
% V = sigma2*eye(p)/(X' * X);
% 
% figure(13)
% hold on
% plot(Nu_data_vals, Res, 'kx', 'linewidth',5)
% plot(Nu_data_vals, 0*ones(n,1),'-b',Nu_data_vals, 2*sigma_hat*ones(n,1),'-r',Nu_data_vals, -2*sigma_hat*ones(n,1),'-r', 'linewidth',3 )
% hold off
% set(gca,'Fontsize',20);
% xlabel('Nu values')
% ylabel('Residuals')
% legend('Residue','','2\sigma interval','Location','Northeast')
% 
% %%
% 
% clear data model options
% 
% data.Re_data = Re_data;
% data.Pr_data = Pr_data;
% data.Nu_data = Nu_data;
% tcov = V;
% N = 50000;
% theta_init = [0.0043036,0.98191,0.40866 ];
% 
% params = {
% {'theta_1',theta_init(1)}
% {'theta_2',theta_init(2)}
% {'theta_3',theta_init(3)}};
% model.ssfun = @SS_fun;
% model.sigma2 = sigma2;
% model.N = n;
% options.qcov = tcov;
% options.nsimu = N;
% options.updatesigma = 1;
% %options.burnin_scale = 10000;
% 
% 
% [results,chain,s2chain] = mcmcrun(model,data,params,options);
% 
% 
% theta_1_vals = chain(10001:end,1);
% theta_2_vals = chain(10001:end,2);
% theta_3_vals = chain(10001:end,3);
% 
% 
% [~,density_theta_1,theta_1_mesh,~]=kde(theta_1_vals);
% [~,density_theta_2,theta_2_mesh,~]=kde(theta_2_vals);
% [~,density_theta_3,theta_3_mesh,~]=kde(theta_3_vals);
% 
% 
% figure(1); clf
% mcmcplot(chain,[],results,'chainpanel');
% 
% figure(2); clf
% mcmcplot(chain,[],results,'pairs');
% 
% cov(chain)
% chainstats(chain,results)
% 
% figure(3); clf
% plot(theta_1_vals,'-','linewidth',2)
% set(gca,'Fontsize',22);
% %axis([0 N 0.009 0.011])
% box on
% xlabel('Chain Iteration')
% ylabel('Parameter \theta_1')
% 
% figure(4); clf
% plot(theta_2_vals,'-','linewidth',2)
% set(gca,'Fontsize',22);
% %axis([0 N 0.009 0.011])
% box on
% xlabel('Chain Iteration')
% ylabel('Parameter \theta_2')
% 
% figure(5); clf
% plot(theta_3_vals,'-','linewidth',2)
% set(gca,'Fontsize',22);
% %axis([0 N 0.009 0.011])
% box on
% xlabel('Chain Iteration')
% ylabel('Parameter \theta_3')
% 
% 
% figure(6); clf
% hold on
% plot(theta_1_mesh,density_theta_1,'k-','linewidth',3)
% set(gca,'Fontsize',22);
% %axis([0.009 0.011  0 3000])
% box on
% xlabel('Parameter \theta_1')
% ylabel("PDF")
% hold off
% 
% figure(7); clf
% hold on
% plot(theta_2_mesh,density_theta_2,'k-','linewidth',3)
% set(gca,'Fontsize',22);
% %axis([0.009 0.011  0 3000])
% box on
% xlabel('Parameter \theta_2')
% ylabel("PDF")
% hold off
% 
% figure(8); clf
% hold on
% plot(theta_3_mesh,density_theta_3,'k-','linewidth',3)
% set(gca,'Fontsize',22);
% %axis([0.009 0.011  0 3000])
% box on
% xlabel('Parameter \theta_3')
% ylabel("PDF")
% hold off
% 
% figure(9); clf
% scatter(theta_1_vals,theta_2_vals)
% box on
% set(gca,'Fontsize',23);
% xlabel('Parameter \theta_1')
% ylabel('Parameter \theta_2')
% %axis([0.009 0.011  0.16 0.24])
% 
% 
% figure(10); clf
% scatter(theta_1_vals,theta_3_vals)
% box on
% set(gca,'Fontsize',23);
% xlabel('Parameter \theta_1')
% ylabel('Parameter \theta_3')
% %axis([0.009 0.011  0.16 0.24])
% 
% 
% figure(11); clf
% scatter(theta_2_vals,theta_3_vals)
% box on
% set(gca,'Fontsize',23);
% xlabel('Parameter \theta_2')
% ylabel('Parameter \theta_3')
% %axis([0.009 0.011  0.16 0.24])
% 
% figure(12)
% plot(s2chain(10001:end))
% hold on
% set(gca,'Fontsize',22);
% %axis([0 N 200 1000])
% xlabel('Chain Iteration')
% ylabel(' \sigma^2')
% 
% mean(s2chain(10001:end))
% 





function SS= SS_fun(params, data)
theta_1 = params(1);     theta_2 = params(2);      theta_3 = params(3);
Re_data = data.Re_data;
Pr_data = data.Pr_data;
Nu_data = data.Nu_data;
Nu = theta_1.*(Re_data.^theta_2).*(Pr_data.^theta_3);
Error = Nu_data - Nu;
SS = Error'*Error;
end


function Error = odeparameterestimation(data,params)
theta_1 = params(1);     theta_2 = params(2);      theta_3 = params(3);
Re_data = data.Re_data;
Pr_data = data.Pr_data;
Nu_data = data.Nu_data;
Nu = theta_1.*(Re_data.^theta_2).*(Pr_data.^theta_3);
Error =  Nu - Nu_data;
end

function Nu = Dittus_Boelter(y,params)
theta_1 = params(1);
theta_2 = params(2);
theta_3 = params(3);
Re = y(:,1);
Pr = y(:,2);
Nu = theta_1.*(Re.^theta_2).*(Pr.^theta_3);
end