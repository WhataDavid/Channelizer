# 读dada文件，提取出极化数据
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import scipy

def load_data(filename):
    ptr = open(filename,"rb")
    file_size = os.path.getsize(filename)
    ptr.seek(4096,0)
    return ptr,file_size

def read_dada(ptr, num):
   
    block_point = 2048 # 2048点
    group_point = block_point * 2 # 4096点

    # bytes_header = 4096 # header=4096bytes
    bytes_per_point = 4 # 每点=4bytes
    bytes_per_block = block_point * bytes_per_point # 每个block=8192bytes 
    bytes_per_group = bytes_per_block * 2 # 每个group=16384bytes

    total_group = num * 2 // group_point # 总组数group=65536
    bytes_total_group = total_group * bytes_per_group # 总组数group的字节数=1073741824bytes
    print("待读取总字节数",bytes_total_group)
    pol1 = np.empty(num, dtype=np.complex64) 
    pol2 = np.empty(num, dtype=np.complex64)

    raw_data = ptr.read(bytes_total_group) # 读取总数据的字节数
    data = np.frombuffer(raw_data, dtype='<i2')
    print("已读取的数据总数data.shape",data.shape)
    
    # 向量化处理offset_binary转换
    # XOR操作:非零值异或0x8000,零值保持为0
    mask = data != 0
    data_converted = data.copy()
    data_converted[mask] ^= 0x8000

    shorts_per_group = bytes_per_group // 2    
    data_reshaped = data_converted.reshape(total_group, shorts_per_group)

    for i in range(total_group):
        group_data = data_reshaped[i]

        # pol1:前2048个复数
        pol1_real = group_data[0:4096:2]
        pol1_imag = group_data[1:4096:2]
        pol1[i*2048:(i+1)*2048] = pol1_real + 1j * pol1_imag
        # pol2:后2048个复数
        pol2_real = group_data[4096::2]
        pol2_imag = group_data[4097::2]
        pol2[i*2048:(i+1)*2048] = pol2_real + 1j * pol2_imag

    return pol1, pol2

def filter_coes(taps, M, D):
    print("======================filtercoes function==========================")

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
    wind = scipy.signal.get_window("hann",N)
    sinc = scipy.signal.firwin(N, cutoff= 1.0 / D, window="hann")
    coe =np.zeros(N, dtype=complex)
    for i in range(coe.shape[0]):
        coe[i] = sinc[i] * wind[i]
    coe1 = np.arange(N)
    # for i in range(coe.shape[0]):
    #     coe[i] *= np.exp(1j * np.pi * coe1[i] / M)
    # print("滤波系数：",coe)
     
    coes = coe.reshape(M, -1, order='F') 
    print("重排后系数矩阵维度：", coes.shape)
    # print("重排后系数矩阵：")
    # print("滤波系数矩阵：", coes)
    print("======================Exit filtercoes function==========================")    
    return coes


def reshape_data(data, M, D, taps, coe):
    print("======================reshape_data and pfb function==========================")

    data = np.array(data, dtype=np.complex64)
    coe = np.array(coe)

    total = M * taps # 一次矩阵运算，所需数据长度
    max_total = (len(data) - 1) // D + 1
    outdata = []

    for i in range(max_total):
        start = i * D
        end = start + total
        data_block = data[start:end]

        if len(data_block) < total: # 数据不够时，填充0
            data_block = np.pad(data_block, (0, total - len(data_block)), mode='constant')

        data1 = data_block.reshape((M,taps),order='F') #数据重排
        data1 = np.fliplr(np.flipud(data1))
        # print(f"第{i}块数据：{data1}")

        data_2 = data1 * coe
        data2 = np.sum(data_2, axis=1) # 卷积过程
        # print(f"第{i}块卷积过程：{data_2}")
        # print(f"第{i}块卷积结果：{data2}")
        outdata.append(data2)

    outdata = np.array(outdata).T
    print(f"最终输出矩阵：/n{outdata.shape}")
    print("======================Exit reshape_data and pfb function==========================")
    return outdata

def cir_data(data, M, D):
    print("======================cir_data function==========================")

    arr_data = data.reshape(M, -1).T
    step = 0
    for i in range(arr_data.shape[0]):  
        if step != 0:
            arr_data[i] = np.roll(arr_data[i], step)  # np.roll 实现循环移位
        step = (step + (M - D)) % M
    outdata = arr_data.T

    print("======================Exit cir_data function==========================")
    return outdata

# 对循环移位输出数据，进行裁切，裁切去掉 N*D/M 个数据点
def cut_data(data, M, D):
    print("======================cut_data function==========================")
    arr_data = np.asanyarray(data)

    arr_data = np.fft.ifft(arr_data, axis=0)
    N = arr_data.shape[1] #原始数据的列数

    data1 = np.fft.fft(arr_data, axis=1)    
    cut = int(N * D / M)
    outdata = data1[:,:cut]

    print("======================Exit cut_data function==========================")
    return outdata               


    ("======================Exit pfb function==========================")
    return cut_outdata

# 对pfb输出的子通道数据(频谱数据)进行频谱绘制，单独绘制每个通道频谱
def plot_sub_all(data, title):

    M = data.shape[0]
    # data = np.fft.ifft(data)
    plt.figure(figsize=(12, 12), constrained_layout=True)
    for i in range(M):
        plt.subplot(M, 1, i + 1)

        # 计算频谱
        freq_data = np.abs(data[i])
        # freq_data = np.abs(np.fft.fft(data[i],axis=0))

        # 计算返回数组中的最大值和最大值的一半
        max_val = np.max(freq_data)
        half_max_val = max_val / 2

        plt.plot(freq_data)
        plt.yticks([0, half_max_val, max_val], [0, 0.5, 1], size=14)
        plt.xlabel('Sampling Points', size=14)
        plt.ylabel('Normalised Amplitude', size=14)
        plt.grid()
        plt.title(f"Channel {i}",size=14)

    # 保存频谱图像
    folder_path = r"./image"
    file_name = title.replace(":", "_").replace("\\", "_").replace("/", "_").replace("?", "_")
    file_path = os.path.join(folder_path,f"{file_name}.svg")
    plt.savefig(file_path)
    plt.show()


def plot_sub(data, CHANNEL_NUM):
    plt.figure(figsize=(12, 16), dpi=400)

    if data.ndim == 2:
        data = np.concatenate((data, np.zeros_like(data)), axis=1)

    for i in range(CHANNEL_NUM):
        plt.subplot(CHANNEL_NUM, 1, i + 1)
        # 调整子图之间的间距
        plt.subplots_adjust(hspace=1)  # 增加垂直间距
        # 计算该通道数据的傅里叶变换
        fft_data = np.abs(np.fft.fft(data[i], axis=0))
        # 绘制频谱
        plt.plot(fft_data)
    # 调整布局，防止子图重叠
    plt.tight_layout()
    plt.show()


def cut_extra_channel_data(data, M, D):
    print("result shape before cut:\n",data.shape)
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


def cut_extra_channel_data_by_mid_v2(data, M, D):
    """在原始FFT排列上裁掉中间频率"""
    data = np.array(data)
    keep_ratio = D / M
    N = data.shape[1]
    cut_amount = int(N * (1 - keep_ratio))
    half_cut = cut_amount // 2

    # 在原始FFT排列中，中间是正频率末尾和负频率开头的交界处
    mid = N // 2
    # 保留前面和后面，裁掉中间
    result_data = np.concatenate((
        data[:, :mid - half_cut],
        data[:, mid + half_cut:]
    ), axis=1)

    return result_data


def cut_extra_channel_data_by_side(data, M, D):
    """从两边切除多余数据，保留中间部分。"""
    data = np.array(data)
    keep_ratio = D / M

    if data.ndim == 1:
        print("cut 1d")
        total_samples = data.shape[0]
        keep_amount = int(total_samples * keep_ratio)
        cut_side_amount = (total_samples - keep_amount) // 2
        start_idx = cut_side_amount
        end_idx = total_samples - cut_side_amount
        result_data = data[start_idx: end_idx]

    elif data.ndim == 2:
        print("cut 2d")
        total_samples = data.shape[1]
        keep_amount = int(total_samples * keep_ratio)
        cut_side_amount = (total_samples - keep_amount) // 2
        start_idx = cut_side_amount
        end_idx = total_samples - cut_side_amount
        result_data = data[:, start_idx: end_idx]

    else:
        raise ValueError(f"Data dimension {data.ndim} not supported.")

    return result_data

# 将所有子通道频谱进行拼接
def reconstruct_spectrum(data, title):
    M = data.shape[0]
    all_spectrum = []

    for i in range(M):
        # sub_spectrum = np.fft.fftshift(np.fft.fft(data[i])) # 输入data为时域数据时，需要进行FFT
        # sub_spectrum = np.fft.fftshift(data[i]) # 输入data为频谱数据时，直接fftshift
        sub_spectrum = np.abs(data[i])
        all_spectrum.append(sub_spectrum)

    all_spectrum = np.concatenate(all_spectrum)

    # 计算返回数组中的最大值和最大值的一半
    max_val = np.max(all_spectrum)
    half_max_val = max_val / 2

    plt.figure(figsize=(12,6), constrained_layout=True)
    plt.plot(all_spectrum)
    plt.yticks([0, half_max_val, max_val], [0, 0.5, 1], size=14)
    plt.xlabel('Sampline Points', size=14)
    plt.ylabel('Normalised Amplitude', size=14)
    plt.grid()
    plt.title(f"{M} Channel Integrated Spectrum",size=14)

    # 保存频谱图像
    folder_path = r"./image"
    file_name = title.replace(":", "_").replace("\\", "_").replace("/", "_").replace("?", "_")
    file_path = os.path.join(folder_path,f"{file_name}.svg")
    plt.savefig(file_path)
    plt.show()


def compute_avg_fft(data, Nfft): #data(4,2^25)
    
    M, N = data.shape
    segments = N // Nfft
    outdata = np.zeros((M, Nfft)) # 输出 (4, Nfft) 的平均 FFT（复数）

    # for i in range(M):
    #     row_data = data[i]
    #     blocks = row_data[:segments*Nfft].reshape(segments, Nfft)   
    #     amp = np.abs(blocks)   
    #     avg_complex = np.mean(amp, axis=0)
    #     outdata[i] = avg_complex

    data = np.fft.ifft(data) # opfb输出数据为频谱数据时
    for i in range(M):
        row_data = data[i]
        blocks = row_data[:segments*Nfft].reshape(segments, Nfft)
        fft_blocks = np.fft.fft(blocks, axis=1)     
        amp = np.abs(fft_blocks)   
        avg_complex = np.mean(amp, axis=0)
        outdata[i] = avg_complex

    return outdata


def add_rfi2(data, channel_num):
    freq = np.fft.fft(data)
    freq_num = freq.shape[0] // channel_num // 2;  # 每个子带有多少点数/2
    for i in range(0, channel_num + 1):
        step = freq_num // (i + 1)
        for j in range(i):
            # freq[freq_num * (i - 1) + step * (j + 1)] = (j+i+1) * 1e7
            # freq[freq_num * (i - 1) + step * (j + 1)] = j+ 1e7
            freq[freq_num * (i - 1) + step * (j + 1)] = 8e6
        for k in range(i):
            freq[freq.shape[0] - freq_num * (i) + step * (k + 1)] = (2*i-k) * 1e7
            # freq[freq.shape[0] - freq_num * (i) + step * (k + 1)] = (i - k - 1) + 1e7
            # freq[freq.shape[0] - freq_num * (i) + step * (k + 1)] = 1e7
    # return np.fft.ifft(freq[0:int((len(freq)/2))])
    return np.fft.ifft(freq)


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
    return data,half_data


def channel_reorder(data):
    data = np.array(data)
    print("channel_reorder data shape:",data.shape)
    mid = data.shape[0] // 2
    print("mid:",mid)
    new_data = np.concatenate((data[mid:], data[:mid]), axis=0)
    return new_data

    return data

def pfb(data, taps, M, D):
    ("======================pfb function==========================")
    pfb_coe = filter_coes(taps, M, D) # 滤波系数重排
    pfb_out = reshape_data(data, M, D, taps, pfb_coe) # 数据重排和卷积过程

    # dx
    ospfb_fft = np.fft.ifft(pfb_out, axis=0)
    plot_sub(ospfb_fft, M)
    cir_outdata = cir_data(pfb_out, M, D)
    ospfb_fft = np.fft.ifft(cir_outdata, axis=0)
    plot_sub(ospfb_fft, M)
    cut_data = cut_extra_channel_data(np.fft.fft(np.fft.ifft(cir_outdata, axis=0)), M, D)
    ospfb_fft = np.fft.ifft(cut_data)
    plot_sub(ospfb_fft, M)
    channel_inner_reorder_reverse_image, half_data = channel_inner_reorder_and_reverse_image(cut_data)
    plot_sub(np.fft.ifft(channel_inner_reorder_reverse_image), M)
    channel_reorder_data = channel_reorder(channel_inner_reorder_reverse_image)
    plot_sub(np.fft.ifft(channel_reorder_data), M)

    # plot_sub(np.fft.ifft(half_data), M)
    # plt.plot(np.abs(half_data.flatten()))
    # plt.show()

    return channel_reorder_data


if __name__ == "__main__":
    filename = "./subband0.dada"
    ptr,file_size = load_data(filename)
    num = 2**20
    pol1, pol2 = read_dada(ptr, num)
    
    taps = 4444
    M = 8
    D = 6
    data = pol1

    rfi_data = add_rfi2(data, M)
    plt.plot(np.abs(np.fft.fftshift(np.fft.fft(rfi_data))))
    plt.show()
    outdata = pfb(rfi_data, taps, M, D)
    outdata = compute_avg_fft(outdata, 2 ** 10)
    plt.plot(outdata[0])
    plt.plot(outdata[1])
    plt.plot(outdata[2])
    plt.plot(outdata[3])
    plt.show()

    plt.plot(np.abs(np.fft.fftshift(np.fft.fft(data))))
    plt.show()
    outdata = pfb(data, taps, M, D)
    # plot_sub_all(outdata, "pfb_subband__spectrum")
    # reconstruct_spectrum(outdata, "pfb_allband_spectrum")
    outdata = compute_avg_fft(outdata, 2**10)
    plt.plot(outdata[0])
    plt.plot(outdata[1])
    plt.plot(outdata[2])
    plt.plot(outdata[3])
    plt.show()