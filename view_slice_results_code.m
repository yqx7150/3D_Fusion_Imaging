a=data2_PA_z60_SNR30_0_1_cut;
figure;  % 创建一个新的图形窗口
% 假设 a 是一个 100x100x100 的三维数组
% 沿第一个维度取最大值
imagesc(squeeze(max(a, [], 1)));  % 使用max函数沿第一个维度取最大值，然后使用squeeze压缩尺寸
colormap('hot');  % 设置颜色图为热图
impixelinfo;  % 在图像窗口中显示像素信息