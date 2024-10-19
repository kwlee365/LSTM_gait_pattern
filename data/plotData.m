clc
clear all
close all

A = readmatrix('result.txt');
B = readtable('./test/test1.csv', 'Format', 'auto');
B = table2array(B);
B_time = [133:1:length(B)]'
figure()
fontsize(100,"points")

subplot(2,1,1)
plot(A(:,1), 'r', LineWidth=2.0)
hold on
plot(A(:,2), 'k--', LineWidth=2.0)
legend(["output", "label"], "FontSize", 25);
xlabel('time(s)',"FontSize",25);
ylabel('percent gait (%)',"FontSize",25)
% subplot(3,1,2)
% plot(B(:,1), LineWidth=2.0)
% xlim([133 length(B)])
% legend('right joint pos')
set(gca, 'FontSize',25);
subplot(2,1,2)
plot(B(:,5), LineWidth=2.0)
hold on
plot(B(:,6), LineWidth=2.0)
plot(B(:,7), LineWidth=2.0)
plot(B(:,8), LineWidth=2.0)
plot(B(:,9), LineWidth=2.0)
plot(B(:,10), LineWidth=2.0)
xlim([133 length(B)])
xlabel('time(s)',"FontSize", 25);
ylabel('imu',"FontSize",25)
legend('right ang x', 'right ang y', 'right ang z', 'right acc x', 'right acc y', 'right acc z', "FontSize", 25)
set(gca, 'FontSize',25);

%%
figure()
data = readmatrix('loss.txt');
data = data(1:263); 
window_size = 3;
filtered_data = movmean(data, window_size);

x = 1:length(data);
orange_color = [230, 159, 0] / 255;  
plot(filtered_data, 'b-', 'LineWidth', 2.0, 'Color', orange_color);
xlabel('Iterations',"FontSize",25);
ylabel('Loss',"FontSize",25);
xlim([1 length(data)])
grid on;
set(gca, 'FontSize',25);
