import numpy as np
import sys
import scipy
import  os
import matplotlib.pyplot as plt

def plot_sub(data, CHANNEL_NUM,D, title,cut=True):
    plt.figure(figsize=(12, 24), dpi=400)
    sub_bangwidth = 800//CHANNEL_NUM

    data = np.concatenate((data, np.zeros_like(data)), axis=1)
    data = np.concatenate((data, np.zeros_like(data)), axis=1)

    for i in range(CHANNEL_NUM):
        plt.subplot(CHANNEL_NUM, 1, i + 1)
        # 调整子图之间的间距
        plt.subplots_adjust(hspace=1)  # 增加垂直间距
        # 计算该通道数据的傅里叶变换
        fft_data = np.abs(np.fft.fftshift(np.fft.fft(data[i], axis=0)))

        # 计算最大值和最大值的一半
        max_val = np.max(fft_data)
        half_max_val = max_val / 2
        # 绘制频谱
        plt.plot(fft_data)
        # 设置 y 轴刻度为最大值的一半和最大值
        plt.yticks([0, half_max_val, max_val], [0, 0.5, 1], size=14)
        max_val = len(fft_data)
        half_max_val = max_val / 2

        x,y = get_denominator(M,D)
        overlap_frequency = int((x/y-1)*400/CHANNEL_NUM*2)
        if cut==True:
            overlap_frequency = 0
        # print("overlap_frequency:",overlap_frequency)
        if CHANNEL_NUM%2==1:
            offset = 0
            if 800 % CHANNEL_NUM != 0:
                offset = 1
            if i < CHANNEL_NUM // 2:
                left_label = 1182 + sub_bangwidth * i
                right_label = 1182 + sub_bangwidth * (i + 1)  +overlap_frequency
                mid_label = (left_label + right_label) // 2
                plt.xticks([0, half_max_val, max_val], [left_label, mid_label, right_label], size=14)
            elif i==CHANNEL_NUM // 2:
                left_label = 1182 + sub_bangwidth * i
                right_label = 1182 + sub_bangwidth * i-overlap_frequency
                mid_label = 1582
                # plt.xticks([0, half_max_val, max_val], [left_label, mid_label, right_label],
                #            size=14)
            else:
                left_label = 1582 - sub_bangwidth * (i - (CHANNEL_NUM // 2))+sub_bangwidth//2
                right_label = 1582 - sub_bangwidth * (i - (CHANNEL_NUM // 2)) - sub_bangwidth - offset+sub_bangwidth//2-overlap_frequency
                mid_label = (left_label + right_label) // 2
                # plt.xticks([0, half_max_val, max_val], [left_label, mid_label, right_label],
                #            size=14)
        if CHANNEL_NUM%2==0:
            offset=0
            if 800%CHANNEL_NUM!=0:
                offset=1
            if i<CHANNEL_NUM//2:
                left_label=1182+sub_bangwidth*i
                right_label=1182+sub_bangwidth*(i+1)+overlap_frequency
                mid_label = (left_label+ right_label)//2
                plt.xticks([0, half_max_val, max_val], [left_label, mid_label, right_label], size=14)
            else:
                left_label=1582 - sub_bangwidth * (i-(CHANNEL_NUM//2))-offset
                right_label=1582 - sub_bangwidth * (i-(CHANNEL_NUM//2))-sub_bangwidth-overlap_frequency-offset
                mid_label = (left_label+ right_label)//2
                # plt.xticks([0, half_max_val, max_val], [left_label, mid_label, right_label],
                #            size=14)
        # 设置坐标轴标签
        plt.xlabel('Frequency(MHz)', size=14)
        plt.ylabel('Normalized Amplitude', size=14)

        # 设置标题
        plt.title(f"Channel {i}",size=14,fontweight='bold')

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

def filtercoes(taps, M, D):
    # 生成滤波系数
    N = M * taps
    # coe = scipy.signal.firwin(N, cutoff=1.0 / D, window="hamming")
    coe = scipy.signal.firwin(N, cutoff=1.0 / D, window=("kaiser", 6))

    print("滤波系数：\n",coe)
    # 保留小数点后7位
    print("保留小数点后7位后的滤波系数:")
    # coe = np.around(coe, decimals=7)
    for i in range(len(coe)):
        print(coe[i],end=',')
    print("\n")
    return coe

def opfb(data, coe,taps, M, D):
    # data = data[:6144]
    # for i in range(0, 1024):
    #     for j in range(0,6):
    #         data[i*6+j] = 0 if i % 2 == 0 else 1
    #         print(data[i*6+j],end=',')

    data = data[:64]
    for i in range(0,64):
        data[i] = i
    print("\n new data:\n",data)
    data = np.array(data)
    # print("data pre10:\n",data[0:10],data.shape)
    data_snake_array = np.zeros(M * taps)
 #    data_snake_array =[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
 # 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0]
    # print(data_snake_array)
    # pre0=1.0
    # pre1=1.0
    pre0 = 0.0
    pre1 = 0.0
    alldata = []
    # print("-------------------------while-------------------")
    while(len(data)>=D):
        for i in range(taps*M-1,M-1,-1):
            data_snake_array[i] = data_snake_array[i-D]
            # print(i,"=",i-D)

        data_snake_array[7]=pre0
        pre0 = data[4]
        data_snake_array[6]=pre1
        pre1 = data[5]

        for i in range(0,6):
            data_snake_array[5-i] = data[i]
        data = data[6:]
        # print("\ndata_snake_array:\n",data_snake_array)
        # print("data pre10:\n",data[0:10],data.shape)

        res = np.zeros(M)

        for i in range(0,M*taps):
            # print(i,":",data_snake_array[i],"*",coe[i],"+",res[i%M])
            res[i%M]+=np.round(data_snake_array[i] * coe[i],7)

        # print(np.round(res,7))
        alldata.append(res)
    # alldata = np.array(alldata)
    print("alldata:\n",alldata)
    flip_data = np.array(alldata).T
    # flip_data = alldata
    # print("flip_data:\n",flip_data,flip_data.shape)
    dx_ospfb_fft = np.fft.ifft(flip_data, axis=0)
    # print("dx_ospfb_fft.shape:", dx_ospfb_fft.shape)
    # print(dx_ospfb_fft)
    plot_sub(dx_ospfb_fft, M, D, "DX " + str(M) + "/" + str(D) + "X ospfb with z gcd and rotate result:", cut=False)

    outdata_cir = cir_data(flip_data, M, D)  # 对pfb卷积输出数据进行循环移位

    # print("py_outdata_cir_8.shape:", outdata_cir.shape)
    # print(outdata_cir)
    # np.savetxt('./csv/py_outdata_cir_8.txt', outdata_cir)

    dx_ospfb_fft = np.fft.ifft(outdata_cir, axis=0)
    # print("dx_ospfb_fft.shape:", dx_ospfb_fft.shape)
    # print(dx_ospfb_fft)
    # np.savetxt('./csv/py_fft_8.txt', dx_ospfb_fft)

    plot_sub(dx_ospfb_fft, M, D, "DX " + str(M) + "/" + str(D) + "X ospfb with z gcd and rotate result:", cut=False)

def cir_data(data, M, D):
    print("======================cir_data function==========================")
    # 0) 若传入的是列表，则转换为 ndarray
    if isinstance(data, (list, tuple)):
        # data 是一个长度为 M 的列表，列表元素应当都是等长的一维 ndarray
        data = np.vstack(data)         # :contentReference[oaicite:2]{index=2}

    # 1) 若无需换向，直接 reshape 并返回 (M, -1)
    if M == D:
        return data.reshape(M, -1)    # reshape 用于改变形状 :contentReference[oaicite:3]{index=3}

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
    return arr_data.T                      # 保留二维数组结构


def opfb_fpga(data,taps,M,D):
    coe = filtercoes(taps, M, D)
    opfb_out = opfb(data,coe,taps,M,D)

# file_path = r'csv/iladata_2507172153.txt'
# # 读取文件内容
# with open(file_path, 'r') as file:
#     lines = file.readlines()
# data = []
# # 逐行处理
# for line in lines:
#     line = line.strip()  # 去除前后空格或换行符
#     if line:  # 确保非空行
#         # 按制表符分割，并将每个16进制数转换为10进制
#         hex_numbers = line.split('\t')
#         dec_numbers = [int(hex_num, 10) for hex_num in hex_numbers]
#         data.extend(dec_numbers)  # 将转换后的10进制数添加到data数组中
# print(data)

# 加载 example_complex.txt 文件并处理
file_path = r'csv\example_complex_87654321.txt'
# 读取文件内容，并转换为 float 类型数组
with open(file_path, 'r') as file:
    data = [complex(line.strip()) for line in file if line.strip()]
# 转换为 numpy 数组以便后续处理
data = np.array(data).real
data =data[:2**19]

Ns = 2 ** 19  # DAC查找表大小
# Ns = 2**16
fs = 2.064e9
# 计算FFT（单边谱）
fft_complex = np.fft.fft(data.real)
freqs_fft = np.fft.fftfreq(Ns, 1 / fs)  # 频率轴（含负频率）
positive_freqs = freqs_fft[:Ns // 2]

# 单边谱（仅正频率有意义）
positive_complex = fft_complex[:Ns // 2]
plt.subplot(1, 1, 1)
plt.plot(positive_freqs, np.abs(positive_complex), 'm')
plt.grid(True)
plt.tight_layout()
plt.show()

if __name__ == "__main__":
    taps = 32
    M = 8
    D = 6
    out_data = opfb_fpga(data, taps, M, D)

    # import datetime
    # seconds = 0xc3a889b6 & 0x3fffffff
    # print(seconds)
    # epoch_start = datetime.datetime(2024, 1, 1, 0, 0, 0)
    # dt = epoch_start + datetime.timedelta(seconds=seconds)
    # print(dt)
