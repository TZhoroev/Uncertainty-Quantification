clear 
close all
c1=2;
c2=1;
theta1 = normrnd(0,1,[1000,1]);
theta2 = normrnd(0,3,[1000,1]);
Y = c1*theta1+c2*theta2;
Y_theta1_mean = zeros(1000,1);
Y_theta2_mean = zeros(1000,1);
for i=1:1000
    Y_theta1_mean(i) = mean(c1*theta1(i)+c2*theta2); % this operation is only valid for matlab. Not for math.
    Y_theta2_mean(i) = mean(c1*theta1+c2*theta2(i));
end
figure(1)
scatter(theta1, Y, "b")
hold on
plot(theta1,Y_theta1_mean, "-r","linewidth",2)
xlim([-10 10])
ylim([-10 10])
xline(0)
yline(0)
hold off

figure(2)

scatter(theta2, Y,"b")
hold on
plot(theta2,Y_theta2_mean,"-r","linewidth",2)
xlim([-10 10])
ylim([-10 10])
xline(0)
yline(0)
hold off


var(Y_theta1_mean)
var(Y_theta2_mean)
