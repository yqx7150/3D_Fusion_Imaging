import scipy.io
import time
import numpy as np
import tensorflow as tf
start_time = time.time()
sigma = 8
radius = 2
n = 5
from keras.models import load_model
def custom_loss(y_true, y_pred):
    key_region_weight = 2.0
    non_key_region_weight = 1.0
    weights = tf.where(tf.equal(y_true, 1), key_region_weight, non_key_region_weight)
    cross_entropy = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    weighted_cross_entropy = cross_entropy * weights
    return tf.reduce_mean(weighted_cross_entropy)

model = load_model('D:\\3D代码文件\\model_output\\weight\\cut3\\weight10\\siamese_model.h5', custom_objects={'custom_loss': custom_loss})



def load_data(mat_file_path):
    data = scipy.io.loadmat(mat_file_path)
    focused_images = data['focusA_datalist']
    defocused_images = data['focusB_datalist']
    return focused_images, defocused_images


def normalize_data(data):
    mean = np.mean(data)
    std = np.std(data)
    normalized_data = (data - mean) / std
    return normalized_data


# 数据不能像数据集那样随机混合
# 遍历每个区块的数据，并做聚焦预测，依照顺序汇总至一维列表
def predict_map(focused_images, defocused_images):
    map_label = []
    for i in range(len(focused_images)):
        focused_t = focused_images[i].reshape(1, 10, 10, 10, 1)
        defocused_t = defocused_images[i].reshape(1, 10, 10, 10, 1)
        prediction = model.predict([focused_t, defocused_t])
        chance_up = prediction[0][0]
        chance_down = prediction[0][1]
        if chance_up < chance_down:
            map_label.append(1)
        else:
            map_label.append(0)
    return map_label


# 由网络得到的一维预测值映射至对应区块
def map_initial(map_label):
    z_points, x_points, y_points = 61, 101, 101
    map_initial = np.zeros(( 61, 101, 101))
    index = 0
    for z in range(n, z_points - n, 2 * n):
        for x in range(n, x_points - n, 2 * n):
            for y in range(n, y_points - n, 2 * n):
                # 根据label值决定是赋值为1还是0
                if map_label[index] == 1:
                    map_initial[z - n: z + n, x - n: x + n, y - n: y + n] = 1
                else:
                    map_initial[z - n: z + n, x - n: x + n, y - n: y + n] = 0
                index += 1  # 移动到下一个label
    return map_initial

# 高斯率滤波
def gaussian_function_3d(x, y, z, sigma):
    return np.exp(-(x ** 2 + y ** 2 + z ** 2) / (2 * sigma ** 2))

def generate_gaussian_kernel(radius, sigma):
    kernel = np.zeros(shape=(2 * radius + 1, 2 * radius + 1, 2 * radius + 1), dtype=float)
    normalization_factor = 0

    for i in range((-1) * radius, radius + 1):
        for j in range((-1) * radius, radius + 1):
            for k in range((-1) * radius, radius + 1):
                kernel[radius + i, radius + j, radius + k] = gaussian_function_3d(i, j, k, sigma)
                normalization_factor += kernel[radius + i, radius + j, radius + k]

    return kernel / normalization_factor


def gaussian_smoothing(decision_map, kernel, radius):
    z_points, x_points, y_points = decision_map.shape
    smoothed_decision_map = np.zeros(shape=(z_points, x_points, y_points), dtype=float)
    kernel_size = kernel.shape

    for z in range(n, z_points - n):
        for x in range(n, x_points - n):
            for y in range(n, y_points - n):
                smoothed_decision_map[z, x, y] = np.sum(np.multiply(kernel, decision_map[z - radius: z + radius + 1,
                                                                            x - radius: x + radius + 1,
                                                                            y - radius: y + radius + 1]))
    return smoothed_decision_map

# 裁剪成与决策图一样大小的进行融合（解决原数据在切块时无法切成整数个单位块所带来的融合问题）
def crop_center(img, crop_shape):
    z, x, y = img.shape
    startz = z // 2 - (crop_shape[0] // 2)
    startx = x // 2 - (crop_shape[1] // 2)
    starty = y // 2 - (crop_shape[2] // 2)
    return img[startz:startz + crop_shape[0], startx:startx + crop_shape[1], starty:starty + crop_shape[2]]


# 加权融合
def weighted_fusion(focus_A, focus_B, decision_map):
    z_points, x_points, y_points = decision_map.shape
    fusion = np.zeros(shape=(z_points, x_points, y_points), dtype=float)
    focus_A_cropped = crop_center(focus_A, decision_map.shape)
    focus_B_cropped = crop_center(focus_B, decision_map.shape)
    fusion = decision_map * focus_A_cropped + (1 - decision_map) * focus_B_cropped
    return fusion


def out_of_focused(focus_A, focus_B, decision_map):
    z_points, x_points, y_points = decision_map.shape
    fusion = np.zeros(shape=(z_points, x_points, y_points), dtype=float)
    fusion = (1 - decision_map) * focus_A + decision_map * focus_B
    return fusion


def volumetric_information_fusion(focus_A, focus_B, sigma, readius, map_initial):
    kernel = generate_gaussian_kernel(radius, sigma)
    final_decision_map = gaussian_smoothing(map_initial, kernel, radius)
    fusion = weighted_fusion(focus_A, focus_B, final_decision_map)
    return fusion


dataNum2 = 2
dataNum = 1
groupNum13 = [13]
groupNum15 = [15, 16]
groupNum = [17]
SNRlevel = [0, 15, 20, 25, 30, 35]
focalPosition1 = [[30, 60], [30, 60], [30, 60], [30, 60]]
focalPosition = [[35, 60], [35, 60], [35, 60], [35, 60], [35, 60], [35, 60], [35, 60]]

for i in range(dataNum):
    fusion_results = {}
    for SNR in SNRlevel:
        data_path = "D:\\3D代码文件\\show_data\dataGroup2\\dataGroup2Fu"+ str(SNR) + ".mat"
        load_data(data_path)
        focused_images, defocused_images = load_data(data_path)
        A = normalize_data(focused_images)
        B = normalize_data(defocused_images)
        map_label = predict_map(A, B)
        map_initial1 = map_initial(map_label)
        data_PA = scipy.io.loadmat(
            "D:\\3D代码文件\\show_data\dataGroup2\\dataGroup2_cut.mat")
        focus_A = data_PA['data2_PA_z' + str(focalPosition[0][0]) + '_SNR' + str(SNR)+'_0_1_cut']
        focus_B = data_PA['data2_PA_z' + str(focalPosition[0][1]) + '_SNR' + str(SNR)+'_0_1_cut']
        fusion = volumetric_information_fusion(focus_A, focus_B, sigma, radius, map_initial1)
        fusion_results['dataGroup2' + str(SNR) ] = fusion
    scipy.io.savemat('D:\\3D代码文件\\model_output\\fusion\\show_data\\dataGroup2fu.mat', fusion_results)


import gc

gc.collect()
print("--- %s seconds for multi-imaging test---" % (time.time() - start_time))