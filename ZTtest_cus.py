import sys

import scipy
import os
import numpy as np
import matplotlib.pyplot as plt

# plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

'''
滤波系数重排函数
'''


def filtercoes(taps, M, D):
    # 测试代码 filtercoes(6, 8, 4)
    # 测试数据
    # coe = []
    # for i in range(taps * M):
    #     coe.append(0 + i)
    # # coe.extend(list(reversed(coe)))
    # coe = np.array(coe)
    # print("滤波系数：",coe)

    # 生成滤波系数
    N = M * taps
    # wind = scipy.signal.get_window("hann", N)
    # sinc = scipy.signal.firwin(N, cutoff=1.0 / D, window="boxcar")
    coe = scipy.signal.firwin(N, cutoff=1.0 / D, window="hamming")
    print(coe)
    # coe = np.zeros(N, dtype=complex)
    # for i in range(coe.shape[0]):
    #     coe[i] = sinc[i] * wind[i]
    # coe1 = np.arange(N)
    # for i in range(coe.shape[0]):
    #     coe[i] *= np.exp(1j * np.pi * coe1[i] / M)

    # dx add code:

    # print("滤波系数：",coe)

    # # 绘制滤波系数频谱图
    # plt.figure(figsize=(12,6))
    # plt.plot(np.abs(np.fft.fft(coe)))
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.title("pfb滤波器系数频谱图")
    # plt.xlabel("频率点")
    # plt.ylabel("幅度")
    # plt.grid(True)
    # plt.show()

    # 按列优先重排为M行的二维矩阵
    coes_reshaped = coe.reshape(M, -1, order='F')  # 'F'表示列优先

    coes = coes_reshaped
    # coes = np.flipud(coes_reshaped) # 上下翻转矩阵

    # 返回重排后的滤波系数
    # print("重排前系数大小：", coes_reshaped.size)
    # print("重排前系数：")
    # print(coe)
    print("重排后系数矩阵维度：", coes.shape)
    # print("重排后系数矩阵：")
    # print(coes)

    for i in range(len(coe)):
        print(coe[i], end=',')
    print("\n-----------------------------")
    for i in range(len(coe)):
        print("{0:.7f}".format(coe[i]), end=',')

    print("\n-----------------------------")
    # 保留小数点后7位
    coes = np.around(coes, decimals=7)
    print("保留小数点后7位后的滤波系数：\n", coes)
    return coes


def reshape_data(data, M, D, taps, coes):
    """
    参数：
        data  : 输入一维数据（list 或 ndarray）
        M     : 通道数
        D     : 步长（降采样因子）
        taps  : 每个通道的抽头数
        coes  : 滤波器系数矩阵，形状为 (M, taps)

    返回：
        data2: 形状为 (M, N) 的输出矩阵，每列是一次滤波运算的输出
    """
    data = np.array(data, dtype=complex)
    coes = np.array(coes)

    assert coes.shape == (M, taps), f"系数矩阵形状必须为 ({M}, {taps})"

    total_required = M * taps
    max_t = (len(data) - 1) // D + 1  # 修改点：确保最后一个数据能被访问
    data2 = []

    for t in range(max_t):
        start = t * D
        end = start + total_required
        data_block = data[start:end]

        # 如果不足 total_required 个数据则补0
        if len(data_block) < total_required:
            pad_width = total_required - len(data_block)
            data_block = np.pad(data_block, (0, pad_width), mode='constant')
            print(f"补0：补充 {pad_width} 个0")

        # 重构为 M x taps，按列优先
        data1 = data_block.reshape((M, taps), order='F')

        # 先列方向上下翻转，再行方向左右翻转
        data1 = np.fliplr(np.flipud(data1))

        # 计算每行加权结果
        out = np.sum(data1 * coes, axis=1)

        # 打印每个步骤
        # print(f"\n--- 第 {t} 步 ---")
        # print(f"提取数据区间：data[{start}:{end}] = {data_block}")
        # print(f"重构为 data1 (shape={data1.shape}):\n{data1}")
        # print(f"滤波器输出（每行加权求和） out:\n{out}")

        data2.append(out)

    data2 = np.array(data2).T  # 转置为 M x N
    print(f"\n最终输出矩阵 data2 (shape={data2.shape}):\n{data2}")
    return data2


'''
每通道滤波系数与数据卷积
'''


def conv(coes, data, M):
    final = []
    for i in range(M):
        convdata = np.convolve(coes[i], data[i])
        final.append(convdata)

    # 返回重排后的数据矩阵
    print("卷积输出数据维度：", (np.array(final).shape))
    # print("卷积输出数据：")
    # print(final)

    return final


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


def plot_spectra(pfb_out):
    """
    输入：二维矩阵 pfb_out
    功能：对每一行计算频谱，并单独绘图显示
    """
    pfb_out = np.array(pfb_out)
    num_rows = pfb_out.shape[0]

    for i in range(num_rows):
        row_data = pfb_out[i, :]

        # 计算频谱
        spectrum = np.fft.fftshift(np.fft.fft(row_data))
        magnitude = 20 * np.log10(np.abs(spectrum) + 1e-12)  # dB，避免log(0)

        # 频率坐标（单位化）
        freqs = np.fft.fftshift(np.fft.fftfreq(len(row_data)))

        # 绘图
        plt.figure()
        plt.plot(freqs, magnitude)
        plt.title(f"Row {i} Spectrum")
        plt.xlabel("Normalized Frequency")
        plt.ylabel("Magnitude (dB)")
        plt.rcParams['font.sans-serif'] = ['Times New Roman']
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# 函数plot_sub，参数data为二维数组，形状(M,N')，第i行对应第i通道时域信号
# def plot_sub(data, CHANNEL_NUM, title):
#     plt.figure(figsize=(12, 12), dpi=400)
#
#     #使用range生成0到CHANNEL_NUM-1，遍历每个通道
#     for i in range(CHANNEL_NUM):
#         plt.subplot(CHANNEL_NUM, 1, i + 1)
#
#         # 对第i通道信号做一维快速傅里叶变换，获得复数频谱，区为负数频谱的模，得到实数数组
#         fft_data = np.abs(np.fft.fft(data[i], axis=0))
#
#         # 计算返回数组中的最大值和最大值的一半
#         max_val = np.max(fft_data)
#         half_max_val = max_val / 2
#
#         # 绘制频谱
#         plt.plot(fft_data)
#
#         # 设置 y 轴刻度为最大值的一半和最大值，y轴标记为0, half_max_val, max_val]
#         # 对应标签[0, 0.5, 1]，标签文字大小14
#         plt.yticks([0, half_max_val, max_val], [0, 0.5, 1], size=14)
#
#         # 设置坐标轴标签
#         plt.xlabel('Sampling Points Number', size=14)
#         plt.ylabel('Normalized Amplitude', size=14)
#
#         # 设置标题
#         #格式化字符串 f"Channel {i}" 指明通道编号，字号 14，字体加粗
#         plt.title(f"Channel {i}",size=14,fontweight='bold')
#
#
#     folder_path = r"C:\Users\dx\Desktop\PFB\pfb\img"
#     # 修改 title，去掉文件名中不允许的字符
#     valid_title = title.replace(":", "_").replace("\\", "_").replace("/", "_").replace("?", "_")
#
#     # 保存图像为svg格式
#     file_path = os.path.join(folder_path, f"{valid_title}.svg")
#     # 调用 Matplotlib 的 tight_layout 自动调整子图间距，防止标签和标题重叠​
#     plt.tight_layout()
#     plt.savefig(file_path,dpi=400)
#     plt.show()

def get_denominator(M, D):
    x, y = M, D
    while D > 0:
        M, D = D, M % D
    x = int(x / M)
    y = int(y / M)
    # print(x, "/", y)
    return x, y


def plot_sub(data, CHANNEL_NUM, D, title, cut=True):
    plt.figure(figsize=(12, 16), dpi=400)
    sub_bangwidth = 800 // CHANNEL_NUM
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
    file_path = os.path.join(folder_path, f"{valid_title}.pdf")
    # 调整布局，防止子图重叠
    plt.tight_layout()
    plt.savefig(file_path, dpi=400)
    plt.show()


def cut_data(data, M, D):
    print("======================cut_data function==========================")
    """
    按尾部截取多相滤波剩余通道的数据：
     - data: 1D 串行流 或 2D (M, N) 矩阵
    """
    arr = np.asanyarray(data)

    # 若是一维数组，则恢复为 (M, -1) 矩阵
    if arr.ndim == 1:
        if arr.size % M != 0:
            raise ValueError(f"长度 {arr.size} 不能整除通道数 {M}")
        arr = arr.reshape(M, -1)  # ↳ 兼容一维输入 :contentReference[oaicite:4]{index=4}

    # 此时 arr.ndim == 2，可安全解包
    _, n_cols = arr.shape
    cut_amount = int(n_cols * D / M)
    print("======================Exit cut_data function==========================")
    return arr[:, :cut_amount]


def pfb(data, taps, M, D):
    pfb_coe = filtercoes(taps, M, D)
    pfb_data = reshape_data(data, M, D, taps, pfb_coe)
    pfb_out = pfb_data

    dx_ospfb_fft = np.fft.ifft(pfb_out, axis=0)
    plot_sub(dx_ospfb_fft, M, D,
             "DX " + str(M) + "/" + str(D) + "X ospfb with z gcd and rotate result:", cut=False)

    # pfb_out = conv(pfb_coe, pfb_data, M)
    outdata_cir = cir_data(pfb_out, M, D)  # 对pfb卷积输出数据进行循环移位

    dx_ospfb_fft = np.fft.ifft(outdata_cir, axis=0)
    plot_sub(dx_ospfb_fft, M, D,
             "DX " + str(M) + "/" + str(D) + "X ospfb with z gcd and rotate result:", cut=False)

    return dx_ospfb_fft


def add_rfi3(data, M, marker_value=1e7):
    """
    在FFT频谱的左右两半按channel_num分组后完全镜像对称地插入固定幅度标记。
    每组内标记等间隔分布，左右中心对称。
    """
    # 1. FFT
    freq = np.fft.fft(data)
    N = freq.size
    half = N // 2
    group = half // M

    # 2. 对每个组插 k 个标记，左右对称
    for k in range(1, M + 1):
        # 左半段：第 k 组范围 [ (k-1)*group,  k*group )
        left_start = (k - 1) * group
        # 组内等分 k+1 段，取第 m 段中点
        for m in range(1, k + 1):
            posL = left_start + m * (group // (k + 1))
            freq[posL] = marker_value
            # 对称位置：右半段中心对称
            posR = (N - posL) % N
            freq[posR] = marker_value

    # 3. IFFT
    return np.fft.ifft(freq)


def add_rfi2(data, channel_num):
    freq = np.fft.fft(data)
    freq_num = freq.shape[0] // channel_num // 2;  # 每个子带有多少点数/2
    for i in range(0, channel_num + 1):
        step = freq_num // (i + 1)
        for j in range(i):
            freq[freq_num * (i - 1) + step * (j + 1)] = (j + i + 1) * 1e4
            # freq[freq_num * (i - 1) + step * (j + 1)] = j+ 1e4
            # freq[freq_num * (i - 1) + step * (j + 1)] = 1e5
        for k in range(i):
            freq[freq.shape[0] - freq_num * (i) + step * (k + 1)] = (2 * i - k) * 1e4
            # freq[freq.shape[0] - freq_num * (i) + step * (k + 1)] = (i - k - 1) + 1e4
            # freq[freq.shape[0] - freq_num * (i) + step * (k + 1)] = 1e5
    return np.fft.ifft(freq)


def add_rfi2_half(data, channel_num):
    freq = np.fft.fft(data)
    freq_num = freq.shape[0] // channel_num;  # 每个子带有多少点数/2
    for i in range(0, (channel_num // 2 + 1)):
        step = freq_num // (i + 1)
        for j in range(i):
            freq[freq_num * (i - 1) + step * (j + 1)] = 1e5
            freq[freq.shape[0] // 2 + freq_num * (i - 1) + step * (j + 1)] = 1e5
            # freq[freq_num * (i - 1) + step * (j + 1)] = j+ 1e4
            # freq[freq_num * (i - 1) + step * (j + 1)] = 1e5
        # for k in range(i):
        #     freq[freq.shape[0] - freq_num * (i) + step * (k + 1)] = (2*i-k) * 1e4
        # freq[freq.shape[0] - freq_num * (i) + step * (k + 1)] = (i - k - 1) + 1e4
        # freq[freq.shape[0] - freq_num * (i) + step * (k + 1)] = 1e5

    return np.fft.ifft(freq)


'''
函数调用
'''
if __name__ == "__main__":
    taps = 4
    M = 8
    D = 6

    data = np.loadtxt(r'C:\Users\dx\Desktop\PFB\pfb\PFB-main\PFB\mini_data.txt')
    data_shifted = np.fft.fftshift(np.abs(np.fft.fft(data)))
    plt.plot(data_shifted)
    plt.show()

    data_shifted = np.fft.fftshift(np.abs(np.fft.fft(data)))
    half_data = len(data_shifted) // 2
    data_spectrum = data_shifted[half_data:]
    plt.plot(data_spectrum)  # 绘制全部频谱
    plt.show()

    # 添加标记信号
    data = add_rfi2_half(data, M)
    # plt.rcParams["font.family"] = ["SimSun", "WenQuanYi Micro Hei", "Heiti TC"]
    plt.figure(figsize=(12, 5), dpi=400)
    plt.yticks([0, 40000, 80000], [0, 0.5, 1], size=14)
    plt.xlabel('采样点数', size=14)
    plt.ylabel('归一化幅度', size=14)
    np_data = np.abs(np.fft.fft(data))
    plt.plot(np_data)  # 绘制全部频谱
    plt.savefig(r'C:\Users\dx\Desktop\PFB\pfb\img\raw_spectrum.svg', format='svg')
    plt.show()

    # 加载 example_complex.txt 文件并处理
    file_path = r'csv\example_complex.txt'
    # 读取文件内容，并转换为 float 类型数组
    with open(file_path, 'r') as file:
        data = [complex(line.strip()) for line in file if line.strip()]
    # 转换为 numpy 数组以便后续处理
    data = np.array(data).real
    print(data)

    Ns = 2 ** 19  # DAC查找表大小
    fs = 2.064e9
    # 计算FFT（单边谱）
    fft_complex = np.fft.fft(data.real)
    freqs_complex = np.fft.fftfreq(Ns, 1 / fs)
    freqs_fft = np.fft.fftfreq(Ns, 1 / fs)  # 频率轴（含负频率）
    positive_freqs = freqs_fft[:Ns // 2]

    # 绘制复数频谱（非对称）
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(freqs_complex, np.abs(fft_complex), 'g')
    plt.grid(True)

    # 单边谱（仅正频率有意义）
    positive_complex = fft_complex[:Ns // 2]
    plt.subplot(1, 2, 2)
    plt.plot(positive_freqs, np.abs(positive_complex), 'm')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    out_data_cut = pfb(data, taps, M, D)
    print(f"输出结果 out_data_cut 的形状为：{out_data_cut.shape}")
    # output = pfb(data, taps, M, D)
    # plot_spectra(output)
    # print("PFB输出结果：")
    # print(output)

    # coes = filtercoes(taps, M, D)
    # output = reshape_data(data, M, D, taps, coes)
    # plot_spectra(output)

    # #输出划分通道后的频谱图
    # fig, subs = plt.subplots(M, 1, figsize=(18, 45))
    # outdata = []
    # for i in range(M):
    #     spectrum = np.abs(np.fft.fft(output[i]))
    #     shifted = np.fft.fftshift(spectrum)
    #     half_len = len(shifted)//2
    #     half_spectrum = shifted[half_len:]
    #     subs[i].plot(half_spectrum)
