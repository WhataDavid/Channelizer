import numpy as np
import sys
import scipy
import os
import matplotlib.pyplot as plt
import psr
import time
from datetime import datetime
import struct

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
        # fft_data = np.abs(np.fft.fft(data[i], axis=0))
        fft_data = np.abs(np.fft.fftshift(np.fft.fft(data[i], axis=0)))

        # 绘制频谱
        plt.plot(fft_data)


        # 设置标题
        plt.title(f"Channel {i}",size=14,fontweight='bold')

    folder_path = r"img/20250106/"
    # 修改 title，去掉文件名中不允许的字符
    valid_title = title.replace(":", "_").replace("\\", "_").replace("/", "_").replace("?", "_")
    # ===================== 新增：添加整张图的总标题 =====================
    plt.suptitle(valid_title,  # 总标题文本（用处理后的合法标题）
                 fontsize=18,  # 总标题字体大小
                 fontweight='bold',  # 加粗
                 y=0.98)  # 调整垂直位置，避免贴顶
    # 给总标题预留顶部空间，防止被tight_layout遮挡
    plt.subplots_adjust(top=0.96)
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
    # coe = scipy.signal.firwin(N, cutoff=1.0 / D, window=("kaiser", 6))

    win_coeffs = scipy.signal.get_window("hamming", taps * M)
    sinc = scipy.signal.firwin(taps * M, cutoff=1.0 / D, window="boxcar")
    coe = np.zeros(win_coeffs.shape[0], dtype=np.complex128)  # 使用更高精度的数据类型
    for i in range(coe.shape[0]):
        coe[i] = sinc[i] * win_coeffs[i]
    nv = np.arange(taps * M)
    for i in range(coe.shape[0]):
        coe[i] *= np.exp(1j * np.pi * nv[i] / M)

    # print("滤波系数：\n",coe)
    # 保留小数点后7位
    # print("保留小数点后7位后的滤波系数:")
    # coe = np.around(coe, decimals=7)
    # for i in range(len(coe)):
    #     print(coe[i],end=',')
    # print("\n")
    return coe


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

def coherent_dedispersion(TAPS, M, D):
    if D == M:
        pfb_type = "cspfb"
    else:
        pfb_type = "ospfb"

    def load_data(filename):
        ptr = open(filename, "rb")
        file_size = os.path.getsize(filename)
        ptr.seek(4096, 0);
        return ptr, file_size

    def subpfb():
        # 获取今天的日期并格式化为字符串
        today_date = datetime.now().strftime('%Y%m%d')

        print("subpfb")
        flag = True

        blocks = pow(2, 25)
        filename = "./data/J0437-4715.dada"

        # 加载数据并跳过头部（4096字节）
        ptr, file_size = load_data(filename)

        # 计算块的数量
        nblock = (file_size - 4096) // (2 * blocks)

        if flag:
            print("\nThe nblock is %d\n" % nblock)

        psize = psr.get_period_size(400.0)
        pdata = np.zeros((psize))
        pnum = np.zeros((psize))
        num = M // 2 + 1
        location = np.zeros((num), dtype=int)
        bw = 400 / (num - 1)

        start_time = time.time()
        # nblock = 10
        for i in range(nblock):
            print("\nThe %d block(s)" % (i + 1))
            if i == 0:
                cur_spend_time = start_time

            byte_count = 2 * blocks
            raw_data = ptr.read(byte_count)

            data = struct.unpack('<' + str(byte_count) + 'b', raw_data)
            pol1 = np.zeros((blocks))
            pol2 = np.zeros((blocks))
            for index in range(blocks // 4):
                pol1[4 * index:4 * (index + 1)] = data[8 * index:8 * index + 4]
                pol2[4 * index:4 * (index + 1)] = data[8 * index + 4:8 * (index + 1)]

            print("read data done, use ", time.time() - cur_spend_time, " seconds, all spend ",
                  time.time() - start_time, " seconds")
            cur_spend_time = time.time()

            if pfb_type == "ospfb":
                print("OSPFB")
                coe = filtercoes(taps, M, D)
                # subfreq1 = opfb(pol1, coe, taps, M, D)
                subfreq1 = opfb_hls_compatible(pol1, coe, taps, M, D)
                # subfreq2 = opfb(pol2, coe, taps, M, D)
                subfreq2 = opfb_hls_compatible(pol2, coe, taps, M, D)
                if i == 0:
                    dx_ospfb_fft_for_integral_pol1 = scipy.fft.fft(subfreq1, axis=1)
                    dx_ospfb_fft_for_integral_pol2 = scipy.fft.fft(subfreq2, axis=1)

                    result_consist_pol1 = []
                    result_consist_pol2 = []

                    for j in range(subfreq1.shape[0]):
                        result_consist_pol1.append(psr.integral_single_channel_zyz(dx_ospfb_fft_for_integral_pol1[j]))
                        result_consist_pol2.append(psr.integral_single_channel_zyz(dx_ospfb_fft_for_integral_pol2[j]))

                    result_consist_pol1 = np.asarray(result_consist_pol1).flatten()
                    result_consist_pol2 = np.asarray(result_consist_pol2).flatten()

                    plt.figure(figsize=(15, 7))
                    plot_data_pol1=np.abs(psr.integral_single_channel_zyz(np.fft.fft(pol1)[:int(np.fft.fft(pol1).size) // 2])[
                                    :int(pol1.size / 2)])
                    plot_data_pol2=np.abs(psr.integral_single_channel_zyz(np.fft.fft(pol2)[:int(np.fft.fft(pol2).size) // 2])[
                                    :int(pol2.size / 2)])
                    plt.plot(plot_data_pol2, label="Original Pol 2")
                    plt.plot(plot_data_pol1, label="Original Pol 1")
                    plt.legend(prop={'size': 14})
                    # plt.yticks([0.0, 1.1e7, 2.2e7], [0, 0.5, 1], size=14)
                    # plt.xticks([0, 16384], [1582, 1182], size=14)
                    plt.xlabel('Frequency (MHz)', size=14)
                    plt.ylabel('Normalized Amplitude', size=14)
                    path = os.path.join("img", today_date, f"original{M}{D}x_{TAPS}tap.jpg")
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    plt.savefig(path, dpi=400)
                    plt.show()

                    plt.figure(figsize=(15, 7))
                    plot_data_pol1 = np.abs(result_consist_pol1)[
                             :int(result_consist_pol1.size / 2)]
                    plot_data_pol2=np.abs(result_consist_pol2)[
                             :int(result_consist_pol2.size / 2)]
                    if M%D==0:
                        plt.plot(plot_data_pol2, label="2x IOSC Pol 2")
                        plt.plot(plot_data_pol1, label="2x IOSC Pol 1")
                    else:
                        plt.plot(plot_data_pol2, label=f"{M}/{D}x ROSC Pol 2")
                        plt.plot(plot_data_pol1, label=f"{M}/{D}x ROSC Pol 1")
                    path1 = os.path.join("txt", today_date, f"ospfb{M}{D}x_pol1_block0_data_{TAPS}tap.txt")
                    path2 = os.path.join("txt", today_date, f"ospfb{M}{D}x_pol2_block0_data_{TAPS}tap.txt")
                    os.makedirs(os.path.dirname(path1), exist_ok=True)
                    np.savetxt(path1, plot_data_pol1)
                    np.savetxt(path2, plot_data_pol2)
                    plt.legend(prop={'size': 14})
                    # plt.yticks([0.0, 0.65e6, 1.3e6], [0, 0.5, 1], size=14)
                    # plt.xticks([0, 16384], [1582, 1182], size=14)
                    plt.xlabel('Frequency (MHz)', size=14)
                    plt.ylabel('Normalized Amplitude', size=14)
                    path = os.path.join("img", today_date, f"ospfb{M}{D}x_{TAPS}tap.jpg")
                    # plt.savefig(path, dpi=400)
                    plt.show()

                    # os._exit()

                print("PFB done, use ", time.time() - cur_spend_time, " seconds, all spend ",
                      time.time() - start_time, " seconds")
                cur_spend_time = time.time()

                psr.coherent_dedispersion_cspfb2(subfreq1, subfreq2, M, pdata, pnum,
                                                 location)
                print("Coherent done, use ", time.time() - cur_spend_time, " seconds, all spend ",
                      time.time() - start_time, " seconds")
                cur_spend_time = time.time()

            elif pfb_type == "cspfb":
                print("CSPFB")

            idata = psr.integral_data_cspfb(pdata, psize, M)
            print("Integral done, use ", time.time() - cur_spend_time, " seconds, all spend ",
                  time.time() - start_time, " seconds")
            cur_spend_time = time.time()

            plt.figure(figsize=(10, 5), dpi=100)
            plt.title("Cur:" + str(i))
            plt.ylabel("Magnitude(dB)")
            plt.xlabel("Phase")
            plt.plot(np.abs(idata))
            plt.show()
            print("Draw sub done, use ", time.time() - cur_spend_time, " seconds, all spend ",
                  time.time() - start_time, " seconds")
            cur_spend_time = time.time()

        ptr.close()

        idata = psr.integral_data_cspfb(pdata, psize, M)

        end = time.time()
        print('time all spend:%s second' % ((end - start_time)))

        if pfb_type == "ospfb":
            np.savetxt(f"txt/{today_date}/ospfb{TAPS}tap.txt", idata)
        else:
            np.savetxt(f"txt/{today_date}/cspfb{TAPS}tap.txt", idata)

        plt.figure(figsize=(10, 5), dpi=100)
        # plt.title("Final")
        plt.ylabel("Magnitude(dB)")
        plt.xlabel("Phase")
        plt.plot(np.abs(idata))
        plt.show()

    subpfb()


def opfb(data, coe,taps, M, D):
    print("opfb start ",datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    # data = data[:6144]
    # for i in range(0, 1024):
    #     for j in range(0,6):
    #         data[i*6+j] = 0 if i % 2 == 0 else 1
    #         print(data[i*6+j],end=',')
    # print("\n new data:\n",data)
    data = np.array(data)
    # print("data pre10:\n",data[0:10],data.shape)
    data_snake_array = np.zeros(M * taps)

    alldata = []
    pre = np.zeros(M - D)
    # print("-------------------------while-------------------")
    while(len(data)>=D):
        # 旧的方法比较笨：
        # for i in range(taps*M-1,M-1,-1):
        #     data_snake_array[i] = data_snake_array[i-D]
        #     # print(i,"=",i-D)
        #
        # for i in range(D, M):
        #     data_snake_array[i] = pre[i - D]
        #     pre[i - D] = data[D - (i - D + 1)]
        #
        # for i in range(0, D):
        #     data_snake_array[D - 1 - i] = data[i]
        # 优化后的新方法：
        # 1. 极简、通用的延迟线移位 (统一向右移动 D 个位置)
        data_snake_array[D:] = data_snake_array[:-D]
        # 2. 填入新的 D 个样本 (逆序填入最前面)
        data_snake_array[:D] = data[:D][::-1]

        data = data[D:]

        # print("\ndata_snake_array:\n",data_snake_array)
        # print("data pre10:\n",data[0:10],data.shape)

        res = np.zeros(M)

        for i in range(0,M*taps):
            # print(i,":",data_snake_array[i],"*",coe[i],"+",res[i%M])
            # res[i%M]+=np.round(data_snake_array[i] * coe[i],7)
            res[i % M] += data_snake_array[i] * coe[i]

        # print(np.round(res,7))
        alldata.append(res)
    # alldata = np.array(alldata)
    flip_data = np.array(alldata).T
    # flip_data = alldata
    # print("flip_data:\n",flip_data,flip_data.shape)
    print("opfb while done ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    # PFB
    # dx_ospfb_fft = np.fft.ifft(flip_data, axis=0)
    # plot_sub(dx_ospfb_fft, M, D, "DX " + str(M) + "/" + str(D) + "X ospfb with z gcd and rotate result:", cut=False)
    # cir
    outdata_cir = cir_data(flip_data, M, D)
    # dx_ospfb_fft = np.fft.ifft(outdata_cir, axis=0)
    print("opfb cir done ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    # plot_sub(dx_ospfb_fft, M, D, "DX " + str(M) + "/" + str(D) + "X ospfb with z gcd and rotate result:", cut=False)
    # cut
    cut_data = cut_extra_channel_data(np.fft.fft(np.fft.ifft(outdata_cir, axis=0)), M, D)
    ospfb_fft = np.fft.ifft(cut_data)
    print("opfb cut done ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    plot_sub(ospfb_fft, M,D,
             "DX " + str(M) + "/" + str(D) + "X ospfb with z gcd and rotate cut result:",cut=True)
    return ospfb_fft

# 算法等价于opfb,但在处理大规模数据时运行速度更快
def opfb_hls_compatible(data, coe, taps, M, D):
    """
    逻辑与 HLS 完全对应，但在 Python 中运行极快
    """
    data_in = np.array(data)
    coe_matrix = np.array(coe).reshape((taps, M))  # 对应 HLS 的 coe[i + j*M]

    # 对应 HLS 的 static float data[TAPS][M]
    # 我们用一个二维数组模拟寄存器组
    reg_file = np.zeros((taps, M))

    alldata = []

    # 模拟 HLS 的 while(1) { I.read() }
    num_frames = len(data_in) // D
    for f in range(num_frames):
        # 1. 模拟输入解析: input_floats[D]
        current_input = data_in[f * D: (f + 1) * D]

        # 2. 模拟移位与更新 (STEP 2)
        # HLS 里是用一个大 for 循环移位，Python 里我们直接 flatten 后切片
        flat_reg = reg_file.flatten()
        flat_reg[D:] = flat_reg[:-D]
        # 插入新数据 (逆序)
        flat_reg[:D] = current_input[::-1]
        # 写回寄存器组结构
        reg_file = flat_reg.reshape((taps, M))

        # 3. 模拟卷积计算 (STEP 3)
        # HLS 里是两层 UNROLL 循环，这里用矩阵乘法等效实现，速度快且结果一致
        # res = sum(data[j][i] * coe[j][i]) over j (taps)
        res = np.sum(reg_file * coe_matrix, axis=0)  # 得到长度为 M 的数组

        alldata.append(res)

    flip_data = np.array(alldata).T
    # print("flip_data:\n",flip_data,flip_data.shape)
    print("opfb while done ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    # PFB
    # dx_ospfb_fft = np.fft.ifft(flip_data, axis=0)
    # plot_sub(dx_ospfb_fft, M, D, "DX " + str(M) + "/" + str(D) + "X ospfb with z gcd and rotate result:", cut=False)
    # cir
    outdata_cir = cir_data(flip_data, M, D)
    # dx_ospfb_fft = np.fft.ifft(outdata_cir, axis=0)
    print("opfb cir done ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    # plot_sub(dx_ospfb_fft, M, D, "DX " + str(M) + "/" + str(D) + "X ospfb with z gcd and rotate result:", cut=False)
    # cut
    cut_data = cut_extra_channel_data(np.fft.fft(np.fft.ifft(outdata_cir, axis=0)), M, D)
    ospfb_fft = np.fft.ifft(cut_data)
    print("opfb cut done ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    # plot_sub(ospfb_fft, M, D,
    #          "DX " + str(M) + "/" + str(D) + "X ospfb with z gcd and rotate cut result:", cut=True)
    return ospfb_fft


if __name__ == "__main__":
    taps = 48
    M = 16
    D = 12

    # 加载 example_complex.txt 文件并处理
    # file_path = r'csv\example_complex_87654321.txt'
    file_path = r'csv\example_complex16-1.txt'
    # 读取文件内容，并转换为 float 类型数组
    with open(file_path, 'r') as file:
        data = [complex(line.strip()) for line in file if line.strip()]
    # 转换为 numpy 数组以便后续处理
    data = np.array(data).real
    data = data[:2 ** 19]

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

    # coe = filtercoes(taps, M, D)
    # opfb_res = opfb(data, coe, taps, M, D)
    coherent_dedispersion(taps, M, D)

    # import datetime
    # seconds = 0xc3a889b6 & 0x3fffffff
    # print(seconds)
    # epoch_start = datetime.datetime(2024, 1, 1, 0, 0, 0)
    # dt = epoch_start + datetime.timedelta(seconds=seconds)
    # print(dt)
