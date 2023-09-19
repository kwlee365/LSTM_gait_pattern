%% Find peak point

clc;
clear all;
close all;

T1 = readtable('walking_data_4km.xlsx', 'Format', 'auto');
T1 = table2array(T1);

q = T1(:,1);
right_acc = [T1(:,2) T1(:,3) T1(:,4)];
right_gyro = [T1(:,5) T1(:,6) T1(:,7)];
left_acc = [T1(:,8) T1(:,9) T1(:,10)];
left_gyro = [T1(:,11) T1(:,12) T1(:,13)];

figure()
hold on
time = [1:size(q)]';
% time = [1:5000]';
plot(time,q(time))
% plot(time,right_acc(time))
% plot(time,right_acc(time))
[TF, P] = islocalmin(q(time),'MinSeparation',70,'SamplePoints',time);
plot(time(TF),q(TF),'*r')

%% Labelling

k = find(TF);
Label = zeros(length(time),1);
for i = 1:1:size(k,1)-1
    for j = k(i):1:k(i+1)
        Label(j) = cast((100 / (k(i+1) - k(i)) * (j-k(i))), "uint8");
    end
end

plot(time,Label)
% legend('q', 'pelv','leg', 'peak')