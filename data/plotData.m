clc
clear all
close all

A= readmatrix('data/plotData_qpos.txt');
B= readmatrix('data/plotData_label.txt');

t = 1:1:size(A,1);
t = t*0.01;

figure()
plot(t, A(:,1), 'b', LineWidth=2)
hold on
plot(t, A(:,2), 'k-.', LineWidth=1.5)
legend(["pred qpos", "target qpos"], "FontSize", 25);
xlabel('time(s)',"FontSize",30);
ylabel('target q(rad)',"FontSize",30)
xlim([0, 10])
set(gca,"FontSize",20)
title("Taget Joint Position Estimation", "FontSize", 40);

figure()
plot(t, A(:,3), 'r', LineWidth=2)
hold on
plot(t, A(:,4), 'g', LineWidth=2)
hold on
plot(t, A(:,5), 'b', LineWidth=2)
legend(["IMU ang x","IMU ang y","IMU ang z"], "FontSize", 25);
xlabel('time(s)',"FontSize",30);
ylabel('Angular vel(rad/s)',"FontSize",30)
xlim([0, 10])
set(gca,"FontSize",20)
title("IMU Angular Velocity", "FontSize", 40);

figure()
plot(t, A(:,6), 'r', LineWidth=2)
hold on
plot(t, A(:,7), 'g', LineWidth=2)
hold on
plot(t, A(:,8), 'b', LineWidth=2)
legend(["IMU acc x","IMU acc y","IMU acc z"], "FontSize", 25);
xlabel('time(s)',"FontSize",30);
ylabel('Linear acc(m/s^2)',"FontSize",30)
xlim([0, 10])
set(gca,"FontSize",20)
title("IMU Linear Acceleration", "FontSize", 40);

t = 1:1:size(B,1);
t = t*0.01;
figure()
plot(t, B(:,1), 'b', LineWidth=2)
hold on
plot(t, B(:,2), 'k-.', LineWidth=1.5)
legend(["pred % gait", "target % gait"], "FontSize", 25);
xlabel('time(s)',"FontSize",30);
ylabel('% gait(%)',"FontSize",30)
xlim([0, 10])
set(gca,"FontSize",20)
title("Percent Gait Estimation", "FontSize", 40);