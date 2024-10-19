%% Find peak point

clc;
clear all;
close all;

dir_name  = '.\data\txt_file\' 
file_name = 'data20.txt'

target_dir = append(dir_name, file_name);

T1 = readtable(target_dir, 'Format', 'auto');
T1 = table2array(T1(:,[1:18]));

right_q = T1(:,1);
left_q  = T1(:,3);
right_imu = T1(:,[5:10]);
left_imu  = T1(:,[11:16]);

figure()
time = [1:size(right_q)]';
% time = [1:5000]';
plot(time, right_q(time))
hold on
% plot(time,right_imu(time, 5))
% plot(time,left_imu(time, 5))
[TF, P] = islocalmin(right_q(time),'MinSeparation',70,'SamplePoints',time);
plot(time(TF), right_q(TF),'*r')

% Labelling

k = find(TF);
Label = zeros(length(time),1);
for i = 1:1:size(k,1)-1
    for j = k(i):1:k(i+1)
        Label(j) = cast((100 / (k(i+1) - k(i)) * (j-k(i))), "uint8");
    end
end

plot(time,Label / 100.0)
% legend('q', 'pelv','leg', 'peak')

%%
start_time = 45
end_time   = 1774
elapsed_time = [start_time:1:end_time]

copy = [T1(elapsed_time,[1:16]) Label(elapsed_time)]