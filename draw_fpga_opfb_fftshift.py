# 000000004066315b00000000cdc0542000000000c031c68e00000000a5e3b72600000000bf8ed4bb00000000c06acff200000000c12761ab000000003ffbc7aa
import sys

import numpy as np
import matplotlib.pyplot as plt
import struct


def hex_to_float(hex_str):
    """将32位十六进制字符串转换为float"""
    # 每8个字符（32bit）对应一个浮点数
    if len(hex_str) % 8 != 0:
        raise ValueError("Hex string length must be multiple of 8")

    float_list = []
    for i in range(0, len(hex_str), 8):
        # 取32位十六进制
        hex_part = hex_str[i:i + 8]
        # 转换为32位无符号整数
        uint_val = int(hex_part, 16)
        # 转换为字节（注意字节序）
        packed = struct.pack('>I', uint_val)  # '>'表示大端序
        # 解析为float
        float_val = struct.unpack('>f', packed)[0]
        # if (i/8)%2==1:
        #     float_list.append(-1*float_val)
        float_list.append(-1 * float_val)
    return float_list


def plot_spectrum(data, sample_rate=1):
    """绘制频谱图"""
    n = len(data)
    y_fft = np.fft.fft(data)
    freq = np.fft.fftfreq(n, 1 / sample_rate)[:n // 2]
    magnitude = 2.0 / n * np.abs(y_fft[:n // 2])

    plt.figure(figsize=(12, 6))
    plt.plot(freq, magnitude)
    plt.title('Frequency Spectrum (Float Data)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.show()


def plot_sub(data, CHANNEL_NUM, D, title, cut=True):
    plt.figure(figsize=(12, 16), dpi=400)
    sub_bangwidth = 800 // CHANNEL_NUM

    if data.ndim == 2:
        data = np.concatenate((data, np.zeros_like(data)), axis=1)
    # data = np.concatenate((data, np.zeros_like(data)), axis=1)

    for i in range(CHANNEL_NUM):
        plt.subplot(CHANNEL_NUM, 1, i + 1)
        # 调整子图之间的间距
        plt.subplots_adjust(hspace=1)  # 增加垂直间距
        # 计算该通道数据的傅里叶变换
        fft_data = np.abs(np.fft.fft(data[i], axis=0))

        # 计算最大值和最大值的一半
        max_val = np.max(fft_data)
        half_max_val = max_val / 2
        # 绘制频谱
        plt.plot(fft_data)
        # 设置 y 轴刻度为最大值的一半和最大值
        plt.yticks([0, half_max_val, max_val], [0, 0.5, 1], size=14)
        max_val = len(fft_data)
        half_max_val = max_val / 2

        x, y = get_denominator(M, D)
        overlap_frequency = int((x / y - 1) * 400 / CHANNEL_NUM * 2)
        if cut == True:
            overlap_frequency = 0
        # print("overlap_frequency:",overlap_frequency)
        if CHANNEL_NUM % 2 == 1:
            offset = 0
            if 800 % CHANNEL_NUM != 0:
                offset = 1
            if i < CHANNEL_NUM // 2:
                left_label = 1182 + sub_bangwidth * i
                right_label = 1182 + sub_bangwidth * (i + 1) + overlap_frequency
                mid_label = (left_label + right_label) // 2
                plt.xticks([0, half_max_val, max_val], [left_label, mid_label, right_label], size=14)
            elif i == CHANNEL_NUM // 2:
                left_label = 1182 + sub_bangwidth * i
                right_label = 1182 + sub_bangwidth * i - overlap_frequency
                mid_label = 1582
                plt.xticks([0, half_max_val, max_val], [left_label, mid_label, right_label],
                           size=14)
            else:
                left_label = 1582 - sub_bangwidth * (i - (CHANNEL_NUM // 2)) + sub_bangwidth // 2
                right_label = 1582 - sub_bangwidth * (
                            i - (CHANNEL_NUM // 2)) - sub_bangwidth - offset + sub_bangwidth // 2 - overlap_frequency
                mid_label = (left_label + right_label) // 2
                plt.xticks([0, half_max_val, max_val], [left_label, mid_label, right_label],
                           size=14)
        if CHANNEL_NUM % 2 == 0:
            offset = 0
            if 800 % CHANNEL_NUM != 0:
                offset = 1
            if i < CHANNEL_NUM // 2:
                left_label = 1182 + sub_bangwidth * i
                right_label = 1182 + sub_bangwidth * (i + 1) + overlap_frequency
                mid_label = (left_label + right_label) // 2
                plt.xticks([0, half_max_val, max_val], [left_label, mid_label, right_label], size=14)
            else:
                left_label = 1582 - sub_bangwidth * (i - (CHANNEL_NUM // 2)) - offset
                right_label = 1582 - sub_bangwidth * (
                            i - (CHANNEL_NUM // 2)) - sub_bangwidth - overlap_frequency - offset
                mid_label = (left_label + right_label) // 2
                plt.xticks([0, half_max_val, max_val], [left_label, mid_label, right_label],
                           size=14)
        # 设置坐标轴标签
        plt.xlabel('Frequency(MHz)', size=14)
        plt.ylabel('Normalized Amplitude', size=14)

        # 设置标题
        plt.title(f"Channel {i}", size=14, fontweight='bold')

    folder_path = r"img/20250106/"
    # 修改 title，去掉文件名中不允许的字符
    valid_title = title.replace(":", "_").replace("\\", "_").replace("/", "_").replace("?", "_")

    # 保存图像
    # file_path = os.path.join(folder_path, f"{valid_title}.pdf")
    # 调整布局，防止子图重叠
    plt.tight_layout()
    # plt.savefig(file_path,dpi=400)
    plt.show()


def get_denominator(M, D):
    x, y = M, D
    while D > 0:
        M, D = D, M % D
    x = int(x / M)
    y = int(y / M)
    # print(x, "/", y)
    return x, y


def cus_roll(data, step):
    N = data.shape[0]
    new_arr = np.empty_like(data)
    for i in range(N):
        idx = (i - step) % N
        new_arr[i] = data[idx]
    return new_arr


def circular_rotate(data, CHANNEL_NUM, D):
    if CHANNEL_NUM == D:
        return data
    np_data = np.reshape(data, (CHANNEL_NUM, -1))
    # print("np_data\n",np_data)
    transpose_data = np.transpose(np_data)
    move_step = 0
    circular_rotate_data = []
    # print("transpose_data\n",transpose_data)
    for i in transpose_data:
        # print("\ni\n", i)
        # print("step\n", move_step)
        if move_step != 0:
            # cur_circular_rotate_data = np.roll(i, move_step)
            cur_circular_rotate_data = cus_roll(i, move_step)
            # print("cur_circular_rotate_data\n", cur_circular_rotate_data)
            # print("cus_circular_rotate_data\n", cus_roll(i, move_step))
            circular_rotate_data.append(cur_circular_rotate_data)
            # print("circular_rotate append done, use ", time.time()  - cur_spend_time, " seconds, all spend ",
            #       time.time() - start_time, " seconds")
            # cur_spend_time = time.time()
        else:
            # print("cur_circular_rotate_data\n", i)
            circular_rotate_data.append(i)
        move_step = (move_step + (CHANNEL_NUM - D)) % CHANNEL_NUM

    circular_rotate_data = np.asarray(circular_rotate_data)
    # print("circular_rotate_data\n", circular_rotate_data)
    circular_rotate_data = np.transpose(circular_rotate_data)
    # print("circular_rotate_data\n", circular_rotate_data)
    # print("circular_rotate result_data.shape:", circular_rotate_data.shape)
    return circular_rotate_data


def cir_data(data, M, D):
    print("======================cir_data function==========================")
    # 0) 若传入的是列表，则转换为 ndarray
    if isinstance(data, (list, tuple)):
        # data 是一个长度为 M 的列表，列表元素应当都是等长的一维 ndarray
        data = np.vstack(data)  # :contentReference[oaicite:2]{index=2}

    # 1) 若无需换向，直接 reshape 并返回 (M, -1)
    if M == D:
        return data.reshape(M, -1)  # reshape 用于改变形状 :contentReference[oaicite:3]{index=3}

    # 2) 将 (M, N) 先转成 (N, M)，方便按时间步循环移位
    arr_data = data.reshape(M, -1).T  # reshape + .T 转置 :contentReference[oaicite:4]{index=4}

    # 3) 按行循环移位模拟 commutator
    step = 0
    for i in range(arr_data.shape[0]):  # 正确迭代行数 :contentReference[oaicite:5]{index=5}
        if step != 0:
            arr_data[i] = np.roll(arr_data[i], step)  # np.roll 实现循环移位 :contentReference[oaicite:6]{index=6}
        step = (step + (M - D)) % M

    print("======================Exit cir_data function==========================")
    # 4) 转置回 (M, N) 并返回
    print((arr_data.T).shape)
    return arr_data.T  # 保留二维数组结构


def cut_extra_channel_data_by_mid(data, M, D):
    print("result shape before cut:\n", data.shape)
    data = np.array(data)
    original_data_rate = D / M
    half_original_data_rate = original_data_rate / 2
    half_original_amount = int(data[0].size * half_original_data_rate)

    # 修改切片逻辑，保留除中间 cut_amount 到 -cut_amount 外的数据
    result_data = np.concatenate((data[:, :half_original_amount], data[:, -half_original_amount:]), axis=1)

    print("cut result_data.shape:\n", result_data.shape)

    # if M == D:
    #     return data
    # data = np.array(data)
    # duplicate_data_rate = D / M
    # cut_amount = int(data[0].size * duplicate_data_rate)
    # result_data = data[:, 0:cut_amount]
    # print("cut result_data.shape:", result_data.shape)

    return result_data


def cut_extra_channel_data_by_side(data, M, D):
    """
    修改后的逻辑：从两边切除多余数据，只保留中间部分。
    """
    print("result shape before cut:\n", data.shape)
    data = np.array(data)

    # 1. 计算原本应该保留的数据比例
    original_data_rate = D / M

    # 2. 计算保留长度的一半
    half_original_data_rate = original_data_rate / 2
    half_original_amount = int(data[0].size * half_original_data_rate)

    # 3. 找到数据的中心点
    mid_point = data.shape[1] // 2

    # 4. 修改切片逻辑：从中心向两边取值 (保留中间)
    # 起点 = 中心 - 半长度
    # 终点 = 中心 + 半长度
    start_idx = mid_point - half_original_amount
    end_idx = mid_point + half_original_amount

    result_data = data[:, start_idx: end_idx]

    print("cut result_data.shape:\n", result_data.shape)
    return result_data


# 示例输入（包含多个32位浮点数）
# hex_input = '4066315bcdc05420c031c68ea5e3b726bf8ed4bbc06acff2c12761ab3ffbc7aa'
# hex_input = '00000000411851ec0000000040970a3d000000004115c28f0000000040947ae10000000041133333000000004091eb84000000004110a3d700000000408f5c28'

# 转换为浮点数列表
# float_data = hex_to_float(hex_input)

# 打印转换结果验证
# print("Converted float values:")
# for i, val in enumerate(float_data):
#     print(f"Index {i}: {val:.6f}")

# 绘制频谱
# plot_spectrum(float_data)

# 加载 iladata_real.txt 文件并处理
# 真实数据：
# file_path = r'csv\iladata_250715016.txt'
# file_path = r'csv\iladata_250717201106_16taps.txt'
# file_path = r'csv\iladata_2507211844_changedacsignal.txt' #改变dac信号12345678后的数据
# file_path = r'csv\iladata_2507221015_sig87654321.txt' #改变dac信号87654321后的数据
file_path = r'csv\iladata_2507232002_dellete0_4tap.txt'
# 测试数据：
# file_path = r'csv\iladata_250717171826.txt'

# 读取文件内容
with open(file_path, 'r') as file:
    lines = file.readlines()

# 初始化 hex_input_array
hex_input_array = []

# 逐行处理
for line in lines:
    line = line.strip()  # 去除前后空格或换行符
    if line:  # 确保非空行
        float_data = hex_to_float(line)
        hex_input_array.append(float_data)

# 输出测试结果
print("Converted float values:")
for i, data in enumerate(hex_input_array):
    print(f"Line {i + 1}: {data}")

M = 8
D = 6
# Line 1: [0.0, -15.750133514404297, 0.0, -17.1505184173584, 0.0, -1.2204030752182007, 0.0, -7.395486831665039, 0.0, 0.24820204079151154, 0.0, 22.585113525390625, 0.0, 6.522176265716553, 0.0, 1.5180305242538452]
reshape_data = np.array(hex_input_array).reshape(len(hex_input_array), 8)
# reshape_data = reshape_data[:, 1::2]

flip_data = np.flip(reshape_data, axis=1)

transpose_data = np.transpose(flip_data)

dx_ospfb_fft = np.fft.ifft(transpose_data, axis=0)
plot_sub(dx_ospfb_fft, M, D, "DX " + str(M) + "/" + str(D) + "X ospfb with z gcd and rotate result:", cut=False)

outdata_cir = cir_data(transpose_data, M, D)  # 对pfb卷积输出数据进行循环移位
dx_ospfb_fft = np.fft.ifft(outdata_cir, axis=0)
plot_sub(dx_ospfb_fft, M, D, "DX " + str(M) + "/" + str(D) + "X ospfb with z gcd and rotate result:", cut=False)

cut_data = cut_extra_channel_data_by_mid(np.fft.fft(np.fft.ifft(outdata_cir, axis=0)), M, D)
plot_sub(np.fft.ifft(cut_data), M, D, "DX " + str(M) + "/" + str(D) + "X ospfb with z gcd and rotate result:")

import queue
import struct
from collections import deque


def channel_reorder_and_reverse_image(data):
    print(data.shape)
    # print(data)
    # print()
    dq = deque()
    for i in range(data.shape[0]):
        dq.append(data[i])
    # dq.pop()
    # dq.pop()
    # for i in dq:
    #     print(i)
    # print(len(dq))
    result = []
    while len(dq) != 0:
        result.append(list(reversed(dq.popleft())))
        result.append(dq.pop())
    # print(result)
    return np.array(result)


# channel_reorder_and_reverse_image = channel_reorder_and_reverse_image(cut_data)
# plot_sub(np.fft.ifft(channel_reorder_and_reverse_image), M, D,"DX " + str(M) + "/" + str(D) + "X ospfb with z gcd and rotate result:")
def channel_reorder(data):
    # print(data.shape)
    # print(data)
    # print()
    dq = deque()
    for i in range(data.shape[0]):
        dq.append(data[i])
    # dq.pop()
    # dq.pop()
    # for i in dq:
    # print(i)
    # print(len(dq))
    result = []
    while len(dq) != 0:
        result.append(dq.popleft())
        result.append(dq.pop())
    # print(result)
    return np.array(result)


# channel_reorder = channel_reorder(cut_data)
# plot_sub(np.fft.ifft(channel_reorder), M, D,"DX " + str(M) + "/" + str(D) + "X ospfb with z gcd and rotate result:")

def cus_roll(data, step):
    N = data.shape[0]
    new_arr = np.empty_like(data)
    for i in range(N):
        idx = (i - step) % N
        new_arr[i] = data[idx]
    return new_arr


def channel_inner_reorder_and_reverse_image(data):
    data = np.array(data)
    print(data.shape)
    # 按行交换前一半和后一半数据
    half = data.shape[1] // 2
    data = np.concatenate((data[:, half:], data[:, :half]), axis=1)
    data = data.flatten()
    data = cus_roll(data, -half)
    half_data = data[:int(data.shape[0] // 2)]
    data = data.reshape(M, -1)
    half_data = half_data.reshape(M, -1)
    print(data.shape)
    return data, half_data


channel_inner_reorder_and_reverse_image, half_data = channel_inner_reorder_and_reverse_image(cut_data)
plot_sub(np.fft.ifft(channel_inner_reorder_and_reverse_image), M, D,
         "DX " + str(M) + "/" + str(D) + "X ospfb with z gcd and rotate result:")
plot_sub(np.fft.ifft(half_data), M, D, "DX " + str(M) + "/" + str(D) + "X ospfb with z gcd and rotate result:")
plt.plot(np.abs(np.concatenate([half_data.flatten(), np.zeros_like(half_data.flatten())], axis=1)))
plt.show()
