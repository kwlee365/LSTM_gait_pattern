%% Find peak point

clc;
clear all;
close all;

% T1 = readtable('walking_data_4kmh_5min_(2).xlsx', 'Format', 'auto');
T1 = readtable('walking_data.xlsx', 'Format', 'auto');
T1 = table2array(T1);

t = T1(:,1);
q = T1(:,2);
pelv_gyro_z = T1(:,6);
leg_gyro_z = T1(:,12);

figure()
hold on
time = [1:size(t)]';
% time = [1:1000]';
plot(time,q(time))
plot(time,pelv_gyro_z(time))
% plot(time,leg_gyro_z(time))
[TF, P] = islocalmin(pelv_gyro_z(time),'MinSeparation',100,'SamplePoints',time);
plot(time(TF),pelv_gyro_z(TF),'*r')

%% Labelling

k = find(TF);
Label = zeros(length(time),1);
for i = 1:1:size(k,1)-1
    for j = k(i):1:k(i+1)
        Label(j) = cast((100 / (k(i+1) - k(i)) * (j-k(i))), "uint8");
    end
end

plot(time,Label)
xlim([0 600])
% legend('q', 'pelv','leg', 'peak')