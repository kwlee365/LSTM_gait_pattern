%% check label

clc;
clear all;
close all;

dir_name  = './test/' 
file_name = 'test1.csv'

target_dir = append(dir_name, file_name);

T1 = readtable(target_dir, 'Format', 'auto');
T1 = table2array(T1);

plot(T1(:,1))
hold on
grid on
plot(T1(:,17) / 100.0)