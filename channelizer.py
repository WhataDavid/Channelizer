import struct
import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.fft import fft
from collections import deque
import psr
import time
from datetime import datetime
import cspfb
import opfb


# 创建 Logger 实例并重定向 sys.stdout
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def reset_save_data():
    np.savetxt('txt/dx_pfb_exe_output.txt', [])
    sys.stdout = Logger('txt/dx_pfb_exe_output.txt')
    # np.savetxt('txt/dx_pfb_cur_first_phase.txt', [])
    # np.savetxt('txt/dx_pfb_cur_hits.txt', [])
    # np.savetxt('txt/dx_pfb_cur_profile.txt', [])
    # np.savetxt('txt/pol1_content.txt', [])
    # np.savetxt('txt/pol2_content.txt', [])
    # np.savetxt('txt/subfreq1_content.txt', [])
    # np.savetxt('txt/subfreq2_content.txt', [])
    # for i in range(CHANNEL_NUM):
    #     np.savetxt('txt/dx_pfb_channel_' + str(i) + '.txt', [])


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

        # # 计算最大值和最大值的一半
        # max_val = np.max(fft_data)
        # half_max_val = max_val / 2
        # # 绘制频谱
        plt.plot(fft_data)
        # # 设置 y 轴刻度为最大值的一半和最大值
        # plt.yticks([0, half_max_val, max_val], [0, 0.5, 1], size=14)
        # max_val = len(fft_data)
        # half_max_val = max_val / 2
        #
        # x,y = get_denominator(M,D)
        # overlap_frequency = int((x/y-1)*400/CHANNEL_NUM*2)
        # if cut==True:
        #     overlap_frequency = 0
        # # print("overlap_frequency:",overlap_frequency)
        # if CHANNEL_NUM%2==1:
        #     offset = 0
        #     if 800 % CHANNEL_NUM != 0:
        #         offset = 1
        #     if i < CHANNEL_NUM // 2:
        #         left_label = 1182 + sub_bangwidth * i
        #         right_label = 1182 + sub_bangwidth * (i + 1)  +overlap_frequency
        #         mid_label = (left_label + right_label) // 2
        #         plt.xticks([0, half_max_val, max_val], [left_label, mid_label, right_label], size=14)
        #     elif i==CHANNEL_NUM // 2:
        #         left_label = 1182 + sub_bangwidth * i
        #         right_label = 1182 + sub_bangwidth * i-overlap_frequency
        #         mid_label = 1582
        #         plt.xticks([0, half_max_val, max_val], [left_label, mid_label, right_label],
        #                    size=14)
        #     else:
        #         left_label = 1582 - sub_bangwidth * (i - (CHANNEL_NUM // 2))+sub_bangwidth//2
        #         right_label = 1582 - sub_bangwidth * (i - (CHANNEL_NUM // 2)) - sub_bangwidth - offset+sub_bangwidth//2-overlap_frequency
        #         mid_label = (left_label + right_label) // 2
        #         plt.xticks([0, half_max_val, max_val], [left_label, mid_label, right_label],
        #                    size=14)
        # if CHANNEL_NUM%2==0:
        #     offset=0
        #     if 800%CHANNEL_NUM!=0:
        #         offset=1
        #     if i<CHANNEL_NUM//2:
        #         left_label=1182+sub_bangwidth*i
        #         right_label=1182+sub_bangwidth*(i+1)+overlap_frequency
        #         mid_label = (left_label+ right_label)//2
        #         plt.xticks([0, half_max_val, max_val], [left_label, mid_label, right_label], size=14)
        #     else:
        #         left_label=1582 - sub_bangwidth * (i-(CHANNEL_NUM//2))-offset
        #         right_label=1582 - sub_bangwidth * (i-(CHANNEL_NUM//2))-sub_bangwidth-overlap_frequency-offset
        #         mid_label = (left_label+ right_label)//2
        #         plt.xticks([0, half_max_val, max_val], [left_label, mid_label, right_label],
        #                    size=14)
        # 设置坐标轴标签
        plt.xlabel('Frequency(MHz)', size=14)
        plt.ylabel('Normalized Amplitude', size=14)

        # 设置标题
        plt.title(f"Channel {i}",size=14,fontweight='bold')

    folder_path = r"img/20250106/"
    # 修改 title，去掉文件名中不允许的字符
    valid_title = title.replace(":", "_").replace("\\", "_").replace("/", "_").replace("?", "_")

    # 保存图像
    file_path = os.path.join(folder_path, f"{valid_title}.pdf")
    # 调整布局，防止子图重叠
    plt.tight_layout()
    plt.savefig(file_path,dpi=400)
    plt.show()


def plot_sub2(data, CHANNEL_NUM, title):
    # data = data[:int(data.size / 2)]  # control single sided spectrum
    plt.figure(figsize=(20, 20))
    for i in range(CHANNEL_NUM):
        plt.subplot(CHANNEL_NUM, 1, i + 1)
        plt.plot(np.abs(data[i]))
        plt.title(title + str(i))
    plt.tight_layout()
    plt.show()


def add_rfi(data, channel_num):
    # freq = np.zeros(data.size, dtype=np.complex64)
    freq = np.fft.fft(data)
    freq_num = freq.shape[0] // channel_num;  # 每个子带有多少点数
    for i in range(1, channel_num + 1):
        step = freq_num // (i + 1)
        for j in range(i):
            freq[freq_num * (i - 1) + step * (j + 1)] = 1e5
    return np.fft.ifft(freq)


def add_rfi2(data, channel_num):
    freq = np.fft.fft(data)
    freq_num = freq.shape[0] // channel_num // 2;  # 每个子带有多少点数/2
    for i in range(0, channel_num + 1):
        step = freq_num // (i + 1)
        for j in range(i):
            # freq[freq_num * (i - 1) + step * (j + 1)] = (j+i+1) * 1e4
            # freq[freq_num * (i - 1) + step * (j + 1)] = j+ 1e4
            freq[freq_num * (i - 1) + step * (j + 1)] = 1e5
        for k in range(i):
            freq[freq.shape[0] - freq_num * (i) + step * (k + 1)] = (2*i-k) * 1e4
            # freq[freq.shape[0] - freq_num * (i) + step * (k + 1)] = (i - k - 1) + 1e4
            # freq[freq.shape[0] - freq_num * (i) + step * (k + 1)] = 1e6
    # return np.fft.ifft(freq[0:int((len(freq)/2))])
    return np.fft.ifft(freq)
    # return np.fft.ifft(freq).real

def conv():
    print("1, 2, 3, 4, 5, 6")
    print("0.3, 0.7, 0.7, 0.3")
    print(np.convolve([1, 2, 3, 4, 5, 6], [0.3, 0.7, 0.7, 0.3]), 'convolve', '\n')

    print("1, 2, 3, 4, 5, 6")
    print("0.3, 0.7, 0.7, 0.3")
    print(np.convolve([1, 2, 3, 4, 5, 6], [0.3, 0.7, 0.7, 0.3], 'same'), 'same', '\n')

    print("1, 2, 3, 4, 5, 6")
    print("0.1, 0.2, 0.3, 0.4")
    print(np.convolve([1, 2, 3, 4, 5, 6, ], [0.1, 0.2, 0.3, 0.4], 'valid'), 'valid', '\n')

    print("1, 2, 3, 4, 5, 6")
    print("0.3, 0.7, 0.7, 0.3")
    print(scipy.signal.lfilter([0.3, 0.7, 0.7, 0.3], 1, [1, 2, 3, 4, 5, 6]), 'lfilter', '\n')


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


def cut_extra_channel_data_by_tail(data, CHANNEL_NUM, D):
    if CHANNEL_NUM == D:
        return data
    data = np.array(data)
    cut_data_rate = D / CHANNEL_NUM
    cut_amount = int(data[0].size * cut_data_rate)
    result_data = data[:, 0:cut_amount]
    # print("cut result_data.shape:", result_data.shape)
    return result_data


def integral_single_channel_zyz(data):
    d = []
    sum = 0
    for i in range(len(data)):
        if i != 0 and i % 1024 == 0:
            d.append(sum)
            sum = 0
        sum += data[i]

    return d


def integral(data):
    final_res = []
    for j in range(data.shape[0]):
        print("j[", j, "]=", data[j].shape)
        cur_res = []
        sum = 0
        for i in range(1, data.shape[1] + 1):
            # print("i[",i,"]=",data[j][i-1])
            if i != 0 and i % 1024 == 0:
                cur_res.append(sum)
                sum = 0
            sum += data[j][i - 1]
        final_res.append(cur_res)

    print("integra_data_shape::", np.array(final_res).shape)
    return np.array(final_res)


def integral_all_channel(data):
    all_data = np.array([])
    all_data = np.append(all_data, data[2][int(len(data[0]) / 2):])
    all_data = np.append(all_data, data[3])
    all_data = np.append(all_data, data[0])
    all_data = np.append(all_data, data[1])
    all_data = np.append(all_data, data[2][:int(len(data[0]) / 2)])
    integral_all_channel_data = integral_single_channel_zyz(np.abs(all_data))
    print("integral_all_channel_data_shape::", np.array(integral_all_channel_data).shape)
    return np.array(integral_all_channel_data)


def get_denominator(M, D):
    x, y = M, D
    while D > 0:
        M, D = D, M % D
    x = int(x / M)
    y = int(y / M)
    # print(x, "/", y)
    return x, y


def realignment_coe(numtaps, M, D):
    # print("======================realignment_coe function==========================")
    coe = []
    for i in range(numtaps*M):
        coe.append(101 + i)
    # coe.extend(list(reversed(coe)))

    # win_coeffs = scipy.signal.get_window("hamming", numtaps * M)
    # sinc = scipy.signal.firwin(numtaps * M, cutoff=1.0 / D, window="boxcar")
    # coe = np.zeros(win_coeffs.shape[0], dtype=np.complex128)  # 使用更高精度的数据类型
    # for i in range(coe.shape[0]):
    #     coe[i] = sinc[i] * win_coeffs[i]
    # nv = np.arange(numtaps * M)
    # for i in range(coe.shape[0]):
    #     coe[i] *= np.exp(1j * np.pi * nv[i] / M)

    coe = scipy.signal.firwin(M*numtaps, cutoff=1.0 / D, window=("kaiser", 6))

    coe_reshape = np.reshape(coe, (M, -1), order='F')
    print(coe_reshape)
    if M == D:
        print("coe do not need append zero")
        return coe_reshape
    else:
        print("coe need add zero")
        x, y = get_denominator(M, D)
        if M / D == int(M / D):
            print("coe add zero directly")
            cur_coll = 1
            # print(coe_reshape.shape[1])
            coe_reshape_add_zero = coe_reshape
            while (cur_coll < coe_reshape_add_zero.shape[1]):
                for i in range(x - 1):
                    coe_reshape_add_zero = np.insert(coe_reshape_add_zero, cur_coll, 0, axis=1)
                    cur_coll += 1;
                cur_coll += 1;
            # coe_reshape_add_zero = np.insert(coe_reshape, 1, 0, axis=1)
            # print(coe_reshape_add_zero)
            return coe_reshape_add_zero
        else:
            print("coe divide to sub filter and add zero")

            # GSC add zero code:
            rows, cols = coe_reshape.shape
            coe_reshape_add_zero = np.zeros((rows, cols * x - x + 1), dtype=coe_reshape.dtype)
            for i in range(rows):
                coe_reshape_add_zero[i, ::x] = coe_reshape[i]
            # print("+++++++++++++++++++++4/3  coe_reshape_add_zero\n", coe_reshape_add_zero)
            # GSC add zero code end
            # print(coe_reshape_add_zero.shape)
            coe_reshape_sub_filter_add_zero = []
            for i in range(coe_reshape_add_zero.shape[0]):
                # print(coe_reshape_add_zero[i])
                # print(coe_reshape_add_zero[i].shape)
                # print(np.reshape(coe_reshape_add_zero[i], (y, -1), order='F'))
                try:
                    l = coe_reshape_add_zero.shape[1]
                    length = (-l) % y
                    coe_reshape_add_zero_len = np.pad(coe_reshape_add_zero, ((0, 0), (0, length)),
                                                                          mode='constant')
                    coe_reshape_sub_filter_add_zero.append(np.reshape(coe_reshape_add_zero_len[i], (y, -1), order='F'))
                except ValueError as e:
                    # print(f"Error occurs: {e}")
                    import tkinter
                    from tkinter import messagebox

                    root = tkinter.Tk()
                    root.withdraw()  # 隐藏主窗口
                    messagebox.showinfo("滤波器抽头数错误提醒", "Please try another TAPS, like multiples of D")
                    assert False, "Please try another TAPS, like multiples of D"
            cur_coll = 1
            print(coe_reshape_sub_filter_add_zero)
            np_coe_reshape_sub_filter_add_zero = np.array(coe_reshape_sub_filter_add_zero)
            # print(np_coe_reshape_sub_filter_add_zero.shape[2])
            # print("np_coe_reshape_sub_filter_add_zero\n", np_coe_reshape_sub_filter_add_zero)
            # print("//////////////////////////////////////reduce dim//////////////////////////////////////")
            np_coe_reshape_sub_filter_add_zero = np_coe_reshape_sub_filter_add_zero.reshape(M * y,
                                                                                            np_coe_reshape_sub_filter_add_zero.shape[
                                                                                                2])
            # print(np_coe_reshape_sub_filter_add_zero)
            # while (cur_coll < np_coe_reshape_sub_filter_add_zero.shape[1]):
            #     for i in range(x - 1):
            #         np_coe_reshape_sub_filter_add_zero = np.insert(np_coe_reshape_sub_filter_add_zero, cur_coll, 0,
            #                                                        axis=1)
            #         cur_coll += 1;
            #     cur_coll += 1;
            # print(np_coe_reshape_sub_filter_add_zero)
            return np_coe_reshape_sub_filter_add_zero


def realignment_data_without_add0(data, channel_num, D):
    # print("======================realignment_data_without_add0 function==========================")
    if data.size % D != 0:
        data = data[:-(data.size % D)]
    # print("data after cut overall data", data)
    res_roll_size = int(data.size / D)
    # print("res_roll_size " + str(res_roll_size))
    #
    # print("res array size " + str(res_roll_size * channel_num))

    polyphase_data = np.zeros(res_roll_size * channel_num, dtype=np.complex64)
    # polyphase_data = np.zeros(int(res_roll_size * channel_num))

    x1 = 0
    y1 = channel_num
    x2 = 0
    y2 = channel_num
    # print(str(x1) + ":" + str(y1) + "=" + str(x2) + ":" + str(y2))
    # print("data[x2:y2]-----------------------0", data[x2:y2])
    polyphase_data[x1:y1] = data[x2:y2]
    # print("polyphase_data[x1:y1]--------------------0", polyphase_data[x1:y1])
    while y2 <= data.size and y1 < res_roll_size * channel_num:
        x1 += channel_num
        y1 += channel_num
        x2 = y2 - (channel_num - D)
        y2 += D
        # print(str(x1) + ":" + str(y1) + "=" + str(x2) + ":" + str(y2), "data.size", data.size)
        if y2 > data.size:
            # print("data[x2:y2]----------------------A")
            # print(data[x2:data.size])
            polyphase_data[x1:y1] = np.concatenate((data[x2:data.size], np.zeros(y2 - data.size)))
            # print("polyphase_data[x1:y1]")
            # print(polyphase_data[x1:y1])
        else:
            # print("data[x2:y2]----------------------B")
            # print(data[x2:y2])
            polyphase_data[x1:y1] = data[x2:y2]
            # print("polyphase_data[x1:y1]")
            # print(polyphase_data[x1:y1])

    # print("polyphase_data after match origin array and new array\n", polyphase_data)
    polyphase_data = polyphase_data.reshape(-1, channel_num)
    # print("polyphase_data before delete zero roll\n", polyphase_data)
    # print("polyphase_data size", polyphase_data.shape[0])
    # print("(polyphase_data.shape[0]*polyphase_data.shape[1])", (polyphase_data.shape[0] * polyphase_data.shape[1]))
    exist_non_zero = any(element != 0 for element in polyphase_data[-1])
    if (not exist_non_zero) and ((polyphase_data.shape[0] * polyphase_data.shape[1]) % channel_num != 0):
        polyphase_data = polyphase_data[:-1]
        # print("cut last line")
    polyphase_data = polyphase_data.reshape((1, -1))
    # print("----------------------------------before oder=F-----------------------------------")
    polyphase_data = polyphase_data.flatten()

    # print(polyphase_data)
    polyphase_data = polyphase_data.reshape((channel_num, -1), order='F')
    # print("----------------------------------before flipud-----------------------------------")
    # print(polyphase_data)
    polyphase_data = np.flipud(polyphase_data)
    # print("-------------------------------------------------------------------------")
    # print("polyphase_data_all\n", polyphase_data)
    # print("======================Exit realignment_data_without_add0 function==========================")
    return polyphase_data


def realignment_data_with_z_gcd(data, channel_num, D):
    # print("======================realignment_data_with_z_gcd function==========================")
    # print("data:\n",data)
    x, y = get_denominator(channel_num, D)
    gcd = math.gcd(channel_num, D)
    # print("gcd=", gcd)
    polyphase_data = np.zeros((channel_num, y, realignment_data_without_add0(data, channel_num, D).shape[1]),
                              dtype=np.complex64)
    # print("polyphase_data:before for\n",polyphase_data)
    for i in range(y):
        # print("\n", i)
        cur_data_wait_realign = np.concatenate((data[i*gcd:], np.zeros(i * gcd)))
        # cur_data_wait_realign = np.concatenate((data[i*M:], np .zeros(i*M)))
        # print(cur_data_wait_realign.shape)
        # print("cur_data_wait_realign:\n",cur_data_wait_realign)
        cur_data = realignment_data_without_add0(cur_data_wait_realign, channel_num, D)
        # print("cur_data_after_realign:\n", cur_data)
        cur_data = np.flipud(cur_data)
        # print("cur_data after flipud:\n",cur_data)
        for j in range(cur_data.shape[0]):
            # print(cur_data.shape, " ", channel_num - j - 1, " ", y - 1 - i, " ", j)
            polyphase_data[channel_num - j - 1][y - 1 - i] = cur_data[j]
        # print("polyphase_data_all\n", polyphase_data)
    # print("\n")
    # print("polyphase_data:after for\n", polyphase_data)
    return polyphase_data


def polyphase_filter_bank_with_denominator_z(data, filter_coeffs, channel_num, D):
    # print("===========================polyphase_filter_bank_with_denominator_z function===============================")
    # print(data)
    M=channel_num
    if M / D == int(M / D):
        # print("filter directly ")
        print(filter_coeffs)
        polyphase_data = realignment_data_without_add0(data, channel_num, D)
        # polyphase_data = np.flipud(polyphase_data)
        # print("polyphase_data after relignment\n", polyphase_data)
        # print("dx polyphase_data after relignment: ", polyphase_data)
        filt_data_conv = []
        # print("-------------------------------------------for-------------------------------")
        for i in range(channel_num):
            print(polyphase_data[i])
            print(filter_coeffs[i])
            # filt_data_conv.append(np.convolve(polyphase_data[i], filter_coeffs[i],'valid'))
            filt_data_conv.append(np.convolve(polyphase_data[i], filter_coeffs[i]))
            # filt_data_conv.append(scipy.signal.lfilter(filter_coeffs[i], 1, polyphase_data[i]))
        # dispatch_data_conv = scipy.fft.ifft(filt_data_conv, axis=0)
        return filt_data_conv
    else:
        # print("filter with sub filter")
        x, y = get_denominator(M, D)
        polyphase_data2 = realignment_data_with_z_gcd(data, channel_num, D)  # control data realignment method. *****
        # polyphase_data = np.flipud(polyphase_data)
        # print("----------------------------------after flipud-----------------------------------")
        # print("polyphase_data1\n", polyphase_data2)
        polyphase_data2 = polyphase_data2.reshape(M * y, polyphase_data2.shape[2])
        # print("\n\npolyphase_data2\n", polyphase_data2)
        # print("polyphase_data2 shape\n", polyphase_data2.shape)
        # sys.exit()
        # print(filter_coeffs)
        filt_data_conv = []
        final_filt_result = []
        start_time = time.time()
        for k in range(channel_num * y):
            # print("\n", k)
            # print("cur polyphase_data_sub:\n", polyphase_data2[k])
            # print("cur filter_coeffs:\n", filter_coeffs[k])
            filt_data_conv.append(np.convolve(polyphase_data2[k], filter_coeffs[k]))
            # filt_data_conv.append(scipy.signal.lfilter(filter_coeffs[k], 1, polyphase_data2[k]))
            # print("filt_data_conv\n", filt_data_conv[k % y])
            print()
            if (k + 1) % y == 0:
                # print("add all array for :\n", filt_data_conv)
                final_filt_result.append(sum(filt_data_conv))
                filt_data_conv.clear()
                # print("final filt result of add all conv result")
                # print(final_filt_result)
        # print("\nall final filt result of add all conv result\n", final_filt_result)
        # dispatch_data_conv = scipy.fft.ifft(final_filt_result, axis=0)
        # print("PFB result_data.shape:", final_filt_result.shape)
        print("filer time:", time.time() - start_time)
        return final_filt_result


def consist_all_subband(dx_ospfb_cut, M):
    consist_result = []

    # 计算每个频谱并将其保存到列表中
    for i in range(M):
        freq_spectrum = np.abs(np.fft.fft(dx_ospfb_cut[i], axis=0))
        consist_result.append(freq_spectrum)
        # plt.plot(freq_spectrum)
        # plt.show()

    # 将所有频谱按顺序拼接在一起
    consist_result_all = np.concatenate(consist_result, axis=0)
    plt.figure(figsize=(8, 8), dpi=400)
    # 绘制拼接后的频谱图
    plt.plot(consist_result_all[:int(len(np.fft.fft(consist_result_all)) / 2)])
    # 计算横坐标最大值和最大值的一半
    max_val = len(consist_result_all)
    half_max_val = max_val / 2
    plt.xticks([0, max_val/2], [1182, 1582], size=14)
    # 计算纵坐标最大值和最大值的一半
    max_val = np.max(consist_result_all)
    half_max_val = max_val / 2

    plt.yticks([0, half_max_val,max_val], [0,0.5,1], size=14)
    plt.xlabel('Frequency(MHz)', size=14)
    plt.ylabel('Normalized Amplitude', size=14)
    # plt.title('Concatenated Spectrum')
    plt.savefig("img/20250106/consist_all_subband.pdf",dpi=400)
    plt.show()

    return consist_result_all

def coherent_dedispersion(TAPS, CHANNEL_NUM, D):
    if D == CHANNEL_NUM:
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
        num = CHANNEL_NUM // 2 + 1
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
                coe = realignment_coe(TAPS, CHANNEL_NUM, D)
                dx_ospfb_out = polyphase_filter_bank_with_denominator_z(pol1, coe, CHANNEL_NUM, D)
                dx_ospfb_rotate = circular_rotate(dx_ospfb_out, CHANNEL_NUM, D)
                dx_ospfb_fft = np.fft.ifft(dx_ospfb_rotate, axis=0)
                dx_ospfb_cut = cut_extra_channel_data_by_tail(np.fft.fft(dx_ospfb_fft), CHANNEL_NUM, D) * D / M
                subfreq1 = np.fft.ifft(dx_ospfb_cut)

                dx_ospfb_out = polyphase_filter_bank_with_denominator_z(pol2, coe, CHANNEL_NUM, D)
                dx_ospfb_rotate = circular_rotate(dx_ospfb_out, CHANNEL_NUM, D)
                dx_ospfb_fft = np.fft.ifft(dx_ospfb_rotate, axis=0)
                dx_ospfb_cut = cut_extra_channel_data_by_tail(np.fft.fft(dx_ospfb_fft), CHANNEL_NUM, D) * D / M
                subfreq2 = np.fft.ifft(dx_ospfb_cut)

                if i == 0:
                    dx_ospfb_fft_for_integral_pol1 = scipy.fft.fft(subfreq1, axis=1)
                    dx_ospfb_fft_for_integral_pol2 = scipy.fft.fft(subfreq2, axis=1)

                    result_consist_pol1 = []
                    result_consist_pol2 = []

                    for j in range(subfreq1.shape[0]):
                        result_consist_pol1.append(integral_single_channel_zyz(dx_ospfb_fft_for_integral_pol1[j]))
                        result_consist_pol2.append(integral_single_channel_zyz(dx_ospfb_fft_for_integral_pol2[j]))

                    result_consist_pol1 = np.asarray(result_consist_pol1).flatten()
                    result_consist_pol2 = np.asarray(result_consist_pol2).flatten()

                    plt.figure(figsize=(15, 7))
                    plot_data_pol1=np.abs(integral_single_channel_zyz(np.fft.fft(pol1)[:int(np.fft.fft(pol1).size) // 2])[
                                    :int(pol1.size / 2)])
                    plot_data_pol2=np.abs(integral_single_channel_zyz(np.fft.fft(pol2)[:int(np.fft.fft(pol2).size) // 2])[
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
                    plt.savefig(path, dpi=400)
                    plt.show()

                    os._exit()

                print("PFB done, use ", time.time() - cur_spend_time, " seconds, all spend ",
                      time.time() - start_time, " seconds")
                cur_spend_time = time.time()

                psr.coherent_dedispersion_cspfb2(subfreq1, subfreq2, CHANNEL_NUM, pdata, pnum,
                                                 location)
                print("Coherent done, use ", time.time() - cur_spend_time, " seconds, all spend ",
                      time.time() - start_time, " seconds")
                cur_spend_time = time.time()

            elif pfb_type == "cspfb":
                print("CSPFB")
                coe = realignment_coe(TAPS, CHANNEL_NUM, D)
                dx_cspfb_out = polyphase_filter_bank_with_denominator_z(pol1, coe, CHANNEL_NUM, D)
                dx_cspfb_fft_subfreq1 = np.fft.ifft(dx_cspfb_out, axis=0)

                dx_cspfb_out = polyphase_filter_bank_with_denominator_z(pol2, coe, CHANNEL_NUM, D)
                dx_cspfb_fft_subfreq2 = np.fft.ifft(dx_cspfb_out, axis=0)

                if i == 0:
                    dx_cspfb_fft_for_integral_pol1 = scipy.fft.fft(dx_cspfb_fft_subfreq1, axis=1)
                    dx_cspfb_fft_for_integral_pol2 = scipy.fft.fft(dx_cspfb_fft_subfreq2, axis=1)

                    result_consist_pol1 = []
                    result_consist_pol2 = []

                    for j in range(dx_cspfb_fft_subfreq1.shape[0]):
                        result_consist_pol1.append(integral_single_channel_zyz(dx_cspfb_fft_for_integral_pol1[j]))
                        result_consist_pol2.append(integral_single_channel_zyz(dx_cspfb_fft_for_integral_pol2[j]))

                    result_consist_pol1 = np.asarray(result_consist_pol1).flatten()
                    result_consist_pol2 = np.asarray(result_consist_pol2).flatten()

                    # draw original figure
                    plt.figure(figsize=(15, 7))
                    plot_data_pol1=np.abs(integral_single_channel_zyz(np.fft.fft(pol1)[:int(np.fft.fft(pol1).size) // 2])[
                                    :int(pol1.size / 2)])
                    plot_data_pol2=np.abs(integral_single_channel_zyz(np.fft.fft(pol2)[:int(np.fft.fft(pol2).size) // 2])[
                                    :int(pol2.size / 2)])
                    plt.plot(plot_data_pol2, label="Original Pol 2")
                    plt.plot(plot_data_pol1, label="Original Pol 1")
                    # np.savetxt("txt/"+today_date+"/original_pol1_block0_data.txt", plot_data_pol1)
                    # np.savetxt("txt/"+today_date+"/original_pol2_block0_data.txt", plot_data_pol2)
                    plt.legend(prop={'size': 14})
                    # plt.yticks([0.0, 1.1e7, 2.2e7], [0, 0.5, 1], size=14)
                    # plt.xticks([0, 16384], [1582, 1182], size=14)
                    plt.xlabel('Frequency (MHz)', size=14)
                    plt.ylabel('Normalized Amplitude', size=14)
                    path = os.path.join("img", today_date, f"original{M}{D}x_{TAPS}tap.jpg")
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    plt.savefig(path, dpi=400)
                    plt.show()

                    # draw Critically sampled channelizer figure
                    plt.figure(figsize=(15, 7))
                    plot_data_pol1=np.abs(result_consist_pol1)[
                             :int(result_consist_pol1.size / 2)]
                    plot_data_pol2=np.abs(result_consist_pol2)[
                             :int(result_consist_pol2.size / 2)]
                    plt.plot(plot_data_pol2, label="CSC Pol 2")
                    plt.plot(plot_data_pol1, label="CSC Pol 1")
                    path1 = os.path.join("txt", today_date, f"cspfb{M}{D}x_pol1_block0_data_{TAPS}tap.txt")
                    path2 = os.path.join("txt", today_date, f"cspfb{M}{D}x_pol2_block0_data_{TAPS}tap.txt")
                    os.makedirs(os.path.dirname(path1), exist_ok=True)
                    np.savetxt(path1, plot_data_pol1)
                    np.savetxt(path2, plot_data_pol2)
                    # plt.yticks([0.0, 0.65e6, 1.3e6], [0, 0.5, 1],size=14)
                    # plt.xticks([0, 16384], [1582, 1182],size=14)
                    plt.xlabel('Frequency (MHz)', size=14)
                    plt.ylabel('Normalized Amplitude', size=14)
                    plt.legend(prop={'size': 14})
                    path = os.path.join("img", today_date, f"cspfb{M}{D}x_{TAPS}tap.jpg")
                    plt.savefig(path, dpi=400)
                    plt.show()

                    os._exit()

                print("PFB done, use ", time.time() - cur_spend_time, " seconds, all spend ",
                      time.time() - start_time, " seconds")
                cur_spend_time = time.time()

                psr.coherent_dedispersion_cspfb2(dx_cspfb_fft_subfreq1, dx_cspfb_fft_subfreq2, CHANNEL_NUM, pdata, pnum,
                                                 location)
                print("Coherent done, use ", time.time() - cur_spend_time, " seconds, all spend ",
                      time.time() - start_time, " seconds")
                cur_spend_time = time.time()

            idata = psr.integral_data_cspfb(pdata, psize, CHANNEL_NUM)
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

        idata = psr.integral_data_cspfb(pdata, psize, CHANNEL_NUM)

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

def compare_pfb(np_data, TAPS, CHANNEL_NUM, D):
    print("----------------------------compare pfb-----------------------------------")
    M=CHANNEL_NUM
    # print("\nZYZ cspfb result:")
    # zyz_cspfb_coe = cspfb.gen_filter_coeffs(TAPS, CHANNEL_NUM)
    # # print("zyzcoe\n", zyz_cspfb_coe)
    # zyz_cspfb_out = cspfb.polyphase_filter(np_data, zyz_cspfb_coe, CHANNEL_NUM)
    # # print("zyz_cspfb_out\n", zyz_cspfb_out)
    # plot_sub(zyz_cspfb_out, CHANNEL_NUM, D,"ZYZ cspfb sub band of ")

    # print("\nZYZ ospfb result:")
    # zyz_ospfb_coe = opfb.gen_filter_coeffs(TAPS, CHANNEL_NUM)
    # print("zyzcoe\n", zyz_ospfb_coe)
    # zyz_ospfb_out = opfb.polyphase_filter(np_data, zyz_ospfb_coe, CHANNEL_NUM)
    # print("zyz_ospfb_out\n", zyz_ospfb_out)
    # plot_sub(zyz_ospfb_out, CHANNEL_NUM, D,"ZYZ ospfb sub band of ")





    dx_ospfb_coe = realignment_coe(TAPS, CHANNEL_NUM, D)
    # print("dxcoe\n", dx_ospfb_coe.real)
    dx_ospfb_out = polyphase_filter_bank_with_denominator_z(np_data, dx_ospfb_coe, CHANNEL_NUM, D)

    dx_ospfb_fft = np.fft.ifft(dx_ospfb_out, axis=0)
    # print(dx_ospfb_fft.shape)
    # print("dx_ospfb_fft\n", dx_ospfb_fft)
    plot_sub(dx_ospfb_fft, CHANNEL_NUM,D, "DX " + str(CHANNEL_NUM) + "/" + str(D) + "X ospfb with z gcd result:",cut=False)

    #
    # print("\nDX " + str(CHANNEL_NUM) + "/" + str(D) + "X ospfb with z gcd and rotate result:")
    dx_ospfb_coe = realignment_coe(TAPS, CHANNEL_NUM, D)
    dx_ospfb_out = polyphase_filter_bank_with_denominator_z(np_data, dx_ospfb_coe, CHANNEL_NUM, D)
    dx_ospfb_rotate = circular_rotate(dx_ospfb_out, CHANNEL_NUM, D)
    dx_ospfb_fft = np.fft.ifft(dx_ospfb_rotate, axis=0)
    # print(dx_ospfb_fft.shape)
    # print("dx_ospfb_fft\n", dx_ospfb_fft)
    plot_sub(dx_ospfb_fft, CHANNEL_NUM,D,
             "DX " + str(CHANNEL_NUM) + "/" + str(D) + "X ospfb with z gcd and rotate result:",cut=False)


    print("\nDX " + str(CHANNEL_NUM) + "/" + str(D) + "X ospfb with z gcd rotate and cut result:")
    dx_ospfb_coe = realignment_coe(TAPS, CHANNEL_NUM, D)
    dx_ospfb_out = polyphase_filter_bank_with_denominator_z(np_data, dx_ospfb_coe, CHANNEL_NUM, D)
    dx_ospfb_rotate = circular_rotate(dx_ospfb_out, CHANNEL_NUM, D)
    dx_ospfb_cut = cut_extra_channel_data_by_tail(np.fft.fft(np.fft.ifft(dx_ospfb_rotate, axis=0)), CHANNEL_NUM,
                                                  D) * D / M
    print(dx_ospfb_cut.shape)
    # print("dx_ospfb_cut\n",dx_ospfb_cut)
    plot_sub(np.fft.ifft(dx_ospfb_cut), CHANNEL_NUM,D,
             "DX " + str(CHANNEL_NUM) + "/" + str(D) + "X ospfb with z gcd and rotate cut result:",cut=True)

    # print("\nDX " + str(CHANNEL_NUM) + "/" + str(D) + "X ospfb with z gcd rotate cut and consist result:")
    # start_time = time.time()
    # dx_ospfb_coe = realignment_coe(TAPS, CHANNEL_NUM, D)
    # # print(dx_ospfb_coe)
    # print("time:", time.time() - start_time)
    # dx_ospfb_out = polyphase_filter_bank_with_denominator_z(np_data, dx_ospfb_coe, CHANNEL_NUM, D)
    # print("time:", time.time() - start_time)
    # dx_ospfb_rotate = circular_rotate(dx_ospfb_out, CHANNEL_NUM, D)
    # dx_ospfb_cut = cut_extra_channel_data_by_tail(np.fft.fft(np.fft.ifft(dx_ospfb_rotate, axis=0)), CHANNEL_NUM,
    #                                               D) * D / M
    # dx_ospfb_consist = consist_all_subband(np.fft.ifft(dx_ospfb_cut),M)
    # print("last")
    # print(dx_ospfb_consist.shape)
    # print(dx_ospfb_consist)


def repeat_frequency():
    np_data = np.loadtxt(r'data/astro_data_104858.txt')
    plt.figure(figsize=(8, 8))
    # 绘制拼接后的频谱图
    plt.plot(abs(np.fft.fft(np_data)))
    plt.show()
    print("np_data.shape", np_data.shape)
    # print(np_data)
    np_data = add_rfi2(np_data, 2)
    print(np_data.shape,np_data)
    cur_frequency_domain = np.fft.fft(np_data)

    cur_frequency_domain[0]=0
    plt.figure(figsize=(8, 8))
    plt.plot(abs(cur_frequency_domain[:int(len(cur_frequency_domain) / 2)]))
    plt.show()

    repeat_result = []
    # 计算每个频谱并将其保存到列表中
    for i in range(5):
        repeat_result.append(cur_frequency_domain)
        # plt.plot(freq_spectrum)
        # plt.show()

    # 将所有频谱按顺序拼接在一起
    repeat_result_all = np.concatenate(repeat_result, axis=0)
    repeat_result_all = repeat_result_all[:-2]
    repeat_result_all = np.fft.ifft(repeat_result_all)

    max_amplitude = max(np.abs(repeat_result_all))  # 找到最大幅度值
    repeat_result_all /= max_amplitude
    repeat_result_all = repeat_result_all.astype(np.complex64)

    plt.figure(figsize=(8, 8))
    # 绘制拼接后的频谱图
    plt.plot(abs(np.fft.fft(repeat_result_all)[:int(len(np.fft.fft(repeat_result_all)) / 2)]))
    plt.show()

    # np.savetxt('data/rfi_data_2g_complex.txt', repeat_result_all)

    cus_data = np.loadtxt('data/rfi_data_2g_complex.txt', dtype=np.complex64)
    plt.figure(figsize=(8, 8))
    # 绘制拼接后的频谱图
    plt.plot(abs(np.fft.fft(cus_data)[:int(len(np.fft.fft(cus_data)) / 2)]))
    plt.show()
    print(cus_data.dtype)
    print(cus_data.shape)



def save_cus_data():
    def load_data(filename):
        ptr = open(filename, "rb")
        file_size = os.path.getsize(filename)
        ptr.seek(4096, 0);
        return ptr, file_size

    blocks = 104858
    print(blocks)
    filename = "./data/J0437-4715.dada"

    # 加载数据并跳过头部（4096字节）
    ptr, file_size = load_data(filename)

    # 计算块的数量
    nblock = (file_size - 4096) // (2 * blocks)
    print(nblock)

    start_time = time.time()
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

        print(pol1.shape)
        # np.savetxt('./data/astro_data_104858.txt', pol1.real)
        sys.exit()


if __name__ == '__main__':
    # conv()
    # filter taps:
    TAPS = 32
    # channel_num(branch), Number of frequency bands :
    CHANNEL_NUM = 16
    # M equal channel_num(branch), more article call it M:
    M = CHANNEL_NUM
    # Decimation factor
    D = 12
    # if D = M : Critical polyphase filter bank(CSPFB)
    # if M is multiples of D : Integer-oversample filter bank(IOSPFB)
    # else if D < M : Rationally-oversampled filter bank(ROSPFB)
    print("----------------------------------start---------------------------------------")
    reset_save_data()

    # 1.测试PFB（支持临界采样、整数倍过采样、分数倍过采样，只需修改通道数CHANNEL_NUM和D抽取因子即可，TAPS为滤波器抽头数）:
    # data = []
    # for i in range(0,12):
    #     data.append(i)
    # np_data = np.array(data)
    # # print(np_data)
    # # circular_rotate(np_data, CHANNEL_NUM,D)
    # # print(np_data)
    # ////////////////////////////////////////////////////////////////////////
    # # 使用真实数据：
    # # 获取当前模块的目录
    # module_dir = os.path.dirname(__file__)
    # # 构建 data.json 文件的完整路径
    # # data_file_path = os.path.join(module_dir, 'mini_data.txt')
    # np_data = np.loadtxt(r'PFB-main\PFB\mini_data.txt')
    # # np_data = np.loadtxt(r'data/astro_data2^19.txt')
    # print(np_data.shape,np_data)
    # plt.plot(np.abs(np.fft.fft(np_data)))
    # plt.show()
    # np_data = add_rfi2(np_data, CHANNEL_NUM)
    # # print(np_data.shape,np_data)
    # # np.savetxt('txt/mini_data_complex.txt', np_data)
    # plt.figure(figsize=(8, 8), dpi=400)
    # # 计算该通道数据的傅里叶变换
    # fft_data = np.abs(np.fft.fft(np_data))
    # # 计算横坐标最大值和最大值的一半
    # # max_val = len(fft_data)
    # # half_max_val = max_val / 2
    # # plt.xticks([0, half_max_val,max_val], [1182,1382,1582], size=14)
    # # # 计算纵坐标最大值和最大值的一半
    # # max_val = np.max(fft_data)
    # # half_max_val = max_val / 2
    # # plt.yticks([0, half_max_val,max_val], [0,0.5,1], size=14)
    # # plt.xlabel('Frequency(MHz)', size=14)
    # # plt.ylabel('Normalized Amplitude', size=14)
    # plt.plot(fft_data)
    # # plt.savefig("img/20250112/before_pfb.pdf",dpi=400)
    # plt.show()
    # //////////////////////////////////////////////////////////////////////////
    # 使用测试数据：
    # 加载 example_complex.txt 文件并处理
    file_path = r'csv\example_complex_87654321.txt'
    # 读取文件内容，并转换为 float 类型数组
    with open(file_path, 'r') as file:
        data = [complex(line.strip()) for line in file if line.strip()]
    # 转换为 numpy 数组以便后续处理
    np_data = np.array(data).real
    np_data = np_data[:2 ** 19]
    plt.plot(np.abs(np.fft.fftshift(np.fft.fft(np_data))))
    plt.show()
    # Ns = 2 ** 19  # DAC查找表大小
    # # Ns = 2**16
    # fs = 2.064e9
    # # 计算FFT（单边谱）
    # fft_complex = np.fft.fft(np_data.real)
    # freqs_fft = np.fft.fftfreq(Ns, 1 / fs)  # 频率轴（含负频率）
    # positive_freqs = freqs_fft[:Ns // 2]
    #
    # # 单边谱（仅正频率有意义）
    # positive_complex = fft_complex[:Ns // 2]
    # plt.subplot(1, 1, 1)
    # plt.plot(positive_freqs, np.abs(positive_complex), 'm')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    # ////////////////////////////////////////////////////////////////////////使用张婷的数据
    # def load_data(filename):
    #     ptr = open(filename, "rb")
    #     file_size = os.path.getsize(filename)
    #     ptr.seek(4096, 0)
    #     return ptr, file_size
    #
    #
    # def read_dada(ptr, num):
    #
    #     block_point = 2048  # 2048点
    #     group_point = block_point * 2  # 4096点
    #
    #     # bytes_header = 4096 # header=4096bytes
    #     bytes_per_point = 4  # 每点=4bytes
    #     bytes_per_block = block_point * bytes_per_point  # 每个block=8192bytes
    #     bytes_per_group = bytes_per_block * 2  # 每个group=16384bytes
    #
    #     total_group = num * 2 // group_point  # 总组数group=65536
    #     bytes_total_group = total_group * bytes_per_group  # 总组数group的字节数=1073741824bytes
    #     print("待读取总字节数", bytes_total_group)
    #     pol1 = np.empty(num, dtype=np.complex64)
    #     pol2 = np.empty(num, dtype=np.complex64)
    #
    #     raw_data = ptr.read(bytes_total_group)  # 读取总数据的字节数
    #     data = np.frombuffer(raw_data, dtype='<i2')
    #     print("已读取的数据总数data.shape", data.shape)
    #
    #     # 向量化处理offset_binary转换
    #     # XOR操作:非零值异或0x8000,零值保持为0
    #     mask = data != 0
    #     data_converted = data.copy()
    #     data_converted[mask] ^= 0x8000
    #
    #     shorts_per_group = bytes_per_group // 2
    #     data_reshaped = data_converted.reshape(total_group, shorts_per_group)
    #
    #     for i in range(total_group):
    #         group_data = data_reshaped[i]
    #
    #         # pol1:前2048个复数
    #         pol1_real = group_data[0:4096:2]
    #         pol1_imag = group_data[1:4096:2]
    #         pol1[i * 2048:(i + 1) * 2048] = pol1_real + 1j * pol1_imag
    #         # pol2:后2048个复数
    #         pol2_real = group_data[4096::2]
    #         pol2_imag = group_data[4097::2]
    #         pol2[i * 2048:(i + 1) * 2048] = pol2_real + 1j * pol2_imag
    #
    #     return pol1, pol2
    #
    #
    # def add_marker(data, fs, marker_freq, marker_amp):
    #     N = len(data)
    #     freq_axis = np.fft.fftfreq(N, 1 / fs)
    #     freq_shifted = np.fft.fftshift(freq_axis)
    #
    #     freq_data = np.fft.fftshift(np.fft.fft(data))
    #
    #     # 3. 处理标记频率（统一为数组格式）
    #     if isinstance(marker_freq, (int, float)):
    #         marker_freq = [marker_freq]
    #
    #     # 4. 遍历每个标记频率，找到最接近的频点索引并设置幅值
    #     for f in marker_freq:
    #         # 找到与目标频率最接近的频点索引
    #         idx = np.argmin(np.abs(freq_shifted - f))
    #         # 设置该频点的幅值（频域值）
    #         freq_data[idx] = marker_amp
    #
    #     # 5. 频域转时域
    #     freq_data = np.fft.ifftshift(freq_data)  # 逆移位
    #     outdata = np.fft.ifft(freq_data)
    #     return outdata
    #
    # def integration(data, Nfft):  # 时域积分：输入data为时域数据(M,N)，输出为频谱outdata(M,Nfft)
    #     datadim = data.ndim == 1
    #     if datadim: data = data.reshape(1, -1)
    #     M, N = data.shape
    #     segment = N // Nfft
    #     outdata = np.zeros((M, Nfft), dtype=np.float64)
    #     for i in range(M):
    #         row_data = data[i]
    #         blocks = row_data[:segment * Nfft].reshape(segment, Nfft)
    #         fft_blocks = np.fft.fft(blocks, axis=1)
    #         amp = np.abs(fft_blocks) ** 2  # 计算功率谱输出
    #         avg = np.mean(amp, axis=0)
    #         outdata[i] = avg
    #     if datadim: outdata = outdata.ravel()
    #     return outdata
    #
    #
    # def plot_freq(data, fs, fc, title):  # data为频谱数据，对频谱直接绘图
    #     n = len(data)
    #     freq = fc + np.fft.fftshift(np.fft.fftfreq(n, d=1 / fs))  # 真实频率轴信息
    #     # amp = 20*np.log10(np.abs(np.fft.fftshift(np.fft.fft(data))))
    #     amp = np.fft.fftshift(np.abs(data))
    #     # plt.figure(figsize=(12,5))
    #     # plt.ylabel('Amplitude', size=14)
    #     # plt.xlabel('Frequency(MHz)', size=14)
    #     # plt.ylabel('幅值', size=14)
    #     # plt.xlabel('采样点数', size=14)
    #     # plt.title(title, size=14)
    #     # plt.plot(freq/1e6, amp)
    #     # plt.plot(amp)
    #
    #     amp_normalized = amp / np.max(amp)  # 计算归一化后的幅值
    #     plt.figure(figsize=(12, 5))
    #     plt.ylabel('Amplitude(normalization)', size=14)
    #     plt.xlabel('Frequency(MHz)', size=14)
    #     plt.plot(freq / 1e6, amp_normalized)
    #     plt.ylim(0, 1.1)
    #     plt.title(title, size=14)
    #     plt.grid(True, alpha=0.3)
    #     plt.show()
    #
    #
    # filename = r'./ZhangTing/subband0.dada'
    # ptr, file_size = load_data(filename)
    # num = 2 ** 20
    # pol1, pol2 = read_dada(ptr, num)
    #
    # data = pol1
    # plt.plot(np.abs(np.fft.fftshift(np.fft.fft(data))))
    # plt.show()
    #
    # f0 = 704e6
    # fc = 768e6
    # fs = 128e6
    # Nfft = 1024
    # fftdata = np.fft.fft(data)
    # plot_freq(fftdata, fs, fc, title="Raw_Data Frequency Spectrum")
    # intrawdata = integration(data, Nfft)
    # plot_freq(intrawdata, fs, fc, title="Raw_Data Frequency Spectrum(nfft=1024)")
    #
    # # 添加RFI
    # # rfidata = addrfi(data,M)
    # rfidata = add_marker(data, fs, marker_freq=[-32e6],
    #                      marker_amp=1e7)  # (-64,-32)(-32,0)(-32.5e6,-31.5e6)(-32.25e6,-31.75e6)
    # fft_rfidata = np.fft.fft(rfidata)
    # plot_freq(fft_rfidata, fs, fc, title="Marker-Data Frequency Spectrum")
    #
    # plt.plot(np.abs(np.fft.fftshift(np.fft.fft(rfidata))))
    # plt.show()
    # ////////////////////////////////////////////////////////////////////////
    compare_pfb(np_data, TAPS, CHANNEL_NUM, D)
    # compare_pfb(fft_rfidata, TAPS, CHANNEL_NUM, D)
    # ////////////////////////////////////////////////////////////////////////
    # 2.PFB并消色散，保存数据到profile中，需关闭测试代码
    # coherent_dedispersion(TAPS, CHANNEL_NUM, D)
    # 3.消色散后profile由当前目录的jupyter文件绘图，注意修改jupyter代码中的profile文件名（subospfb.txt/subcspfb.txt）
    # plot_profile.ipynb
    # plot_sub.ipynb
    # plot_block_dtw.ipynb
    print("--------------------------------------------end------------------------------------------------------")
    # save_cus_data()
    # repeat_frequency()