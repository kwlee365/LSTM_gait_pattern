clc
clear all
close all

A= readmatrix('data/plotData.txt');

t = 1:1:size(A,1);
t = t*0.01;
figure()
plot(t, A(:,1), 'b', LineWidth=1.5)
hold on
plot(t, A(:,2), 'k-.', LineWidth=2)
legend(["pred % gait", "target % gait"], "FontSize", 25);
xlabel('time(s)',"FontSize",30);
ylabel('% gait(%)',"FontSize",30)
xlim([0, 10])
set(gca,"FontSize",20)
title("Percent Gait Estimation", "FontSize", 30);
