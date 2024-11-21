import queue
import struct
import sys
import time
from math import ceil
import pylab as pl
import os
import cspfb
import opfb
import matplotlib
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.fft import fft
from collections import deque
import multiprocessing
import psr
import time
import threading


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


def plot_sub(data, CHANNEL_NUM, title):
    plt.figure(figsize=(12, 12), dpi=400)

    for i in range(CHANNEL_NUM):
        plt.subplot(CHANNEL_NUM, 1, i + 1)

        # 计算该通道数据的傅里叶变换
        fft_data = np.abs(np.fft.fft(data[i], axis=0))

        # 计算最大值和最大值的一半
        max_val = np.max(fft_data)
        half_max_val = max_val / 2

        # 绘制频谱
        plt.plot(fft_data)

        # 设置 y 轴刻度为最大值的一半和最大值
        plt.yticks([0, half_max_val, max_val], [0, 0.5, 1], size=14)

        # 设置坐标轴标签
        plt.xlabel('Sampling Points Number', size=14)
        plt.ylabel('Normalized Amplitude', size=14)

        # 设置标题
        plt.title(f"Channel {i}",size=14,fontweight='bold')

    folder_path = r"img/imag_after_cutff/1117/"
    # 修改 title，去掉文件名中不允许的字符
    valid_title = title.replace(":", "_").replace("\\", "_").replace("/", "_").replace("?", "_")

    # 保存图像
    file_path = os.path.join(folder_path, f"{valid_title}.jpg")
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
            freq[freq_num * (i - 1) + step * (j + 1)] = (j+i+1) * 1e4
            # freq[freq_num * (i - 1) + step * (j + 1)] = j+ 1e4
            # freq[freq_num * (i - 1) + step * (j + 1)] = 1e5
        for k in range(i):
            freq[freq.shape[0] - freq_num * (i) + step * (k + 1)] = (2*i-k) * 1e4
            # freq[freq.shape[0] - freq_num * (i) + step * (k + 1)] = (i - k - 1) + 1e4
            # freq[freq.shape[0] - freq_num * (i) + step * (k + 1)] = 1e5
    return np.fft.ifft(freq)


def gen_filter_coeffs(numtaps, M, D):
    coeffs = scipy.signal.firwin(numtaps * M, cutoff=1.0 / D, window="hamming")
    coeffs = np.reshape(coeffs, (M, -1), order='F')
    return coeffs


def conv():
    # data = []
    # for i in range(1, 5):
    #     data.append(i)
    # np_data = np.asarray(data)
    # reshape_data = np.reshape(np_data, (4, -1), order='F')
    # polyphase_data = np.flipud(reshape_data)
    # print("polyphase " + str(polyphase_data))
    # coe = []
    # for i in range(101, 103):
    #     coe.append(i)
    # coe.extend(list(reversed(coe)))
    # np_coe = np.asarray(coe)
    # coeffs = np.reshape(np_coe, (4, -1), order='F')
    # print("coeffs " + str(coeffs))
    filterd_signals = []
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
        # print("i\n", i)
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


def cut_extra_channel_data(data, CHANNEL_NUM, D):
    if CHANNEL_NUM == D:
        return data
    duplicate_data_rate = 1 - D / CHANNEL_NUM
    cut_rate_per_direction = duplicate_data_rate / 2
    data = np.array(data)
    cut_amount = int(data[0].size * cut_rate_per_direction)
    result_data = data[:, cut_amount:-cut_amount]
    return result_data


def cut_extra_channel_data_by_front(data, CHANNEL_NUM, D):
    if CHANNEL_NUM == D:
        return data
    data = np.array(data)
    duplicate_data_rate = D / CHANNEL_NUM
    cut_amount = int(data[0].size * duplicate_data_rate)
    result_data = data[:, -cut_amount:]
    print("cut result_data.shape:", result_data.shape)
    return result_data[:, -int(result_data.shape[1] / 2):]


def cut_extra_channel_data_by_tail(data, CHANNEL_NUM, D):
    if CHANNEL_NUM == D:
        return data
    data = np.array(data)
    cut_data_rate = D / CHANNEL_NUM
    cut_amount = int(data[0].size * cut_data_rate)
    result_data = data[:, 0:cut_amount]
    # print("cut result_data.shape:", result_data.shape)
    return result_data


# def cut_extra_channel_data_by_tail_of_timedomain(data, CHANNEL_NUM, D):
#     if CHANNEL_NUM == D:
#         return data
#     data = np.fft.fft(np.fft.ifft(data, axis=0))
#     plot_sub(np.fft.ifft(data), CHANNEL_NUM,
#              "DX " + str(CHANNEL_NUM) + "/" + str(D) + "X ospfb with z gcd and rotate cut result:")
#     data = np.array(data)
#     duplicate_data_rate = D / CHANNEL_NUM
#     cut_amount = int(data[0].size * duplicate_data_rate)
#     result_data = data[:, 0:cut_amount]
#     print("cut result_data.shape:", result_data.shape)
#     return result_data


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


def channel_inner_reorder(data):
    result = np.array([])
    for i in range(data.shape[0]):
        result = np.append(result, data[i])
    result = cus_roll(result, -int(len(data[0]) / 2))
    return result


def channel_reorder(data):
    print(data.shape)
    print(data)
    print()
    dq = deque()
    for i in range(data.shape[0]):
        dq.append(data[i])
    # dq.pop()
    # dq.pop()
    for i in dq:
        print(i)
    print(len(dq))
    result = []
    while len(dq) != 0:
        result.append(dq.popleft())
        result.append(dq.pop())
    # print(result)
    return np.array(result)


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
    print(x, "/", y)
    return x, y


def realignment_coe(numtaps, M, D):
    print("======================realignment_coe function==========================")
    # coe = []
    # for i in range(numtaps*M):
    #     coe.append(100 + i)
    # coe.extend(list(reversed(coe)))

    # coe = scipy.signal.firwin(numtaps * M, cutoff=1.0 / D, window="hamming")

    win_coeffs = scipy.signal.get_window("boxcar", numtaps * M)
    sinc = scipy.signal.firwin(numtaps * M, cutoff=1.0 / D, window="boxcar")
    coe = np.zeros(win_coeffs.shape[0], dtype=complex)
    for i in range(coe.shape[0]):
        coe[i] = sinc[i] * win_coeffs[i]
    nv = np.arange(numtaps * M)
    for i in range(coe.shape[0]):
        coe[i] *= np.exp(1j * np.pi * nv[i] / M)

    # print("coe+++++++++++++++++++\n", coe)
    coe_reshape = np.reshape(coe, (M, -1), order='F')
    print(coe_reshape)
    if M == D:
        print("do not need append zero")
        return coe_reshape
    else:
        print("need add zero")
        x, y = get_denominator(M, D)
        if M / D == int(M / D):
            print("add zero directly")
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
            print("divide to sub filter and add zero")

            # GSC add zero code:
            rows, cols = coe_reshape.shape
            coe_reshape_add_zero = np.zeros((rows, cols * x - x + 1), dtype=coe_reshape.dtype)
            for i in range(rows):
                coe_reshape_add_zero[i, ::x] = coe_reshape[i]
            print("+++++++++++++++++++++4/3  coe_reshape_add_zero\n", coe_reshape_add_zero)
            # GSC add zero code end

            coe_reshape_sub_filter_add_zero = []
            for i in range(coe_reshape_add_zero.shape[0]):
                # print(coe_reshape_add_zero[i])
                # print(coe_reshape_add_zero[i].shape)
                # print(np.reshape(coe_reshape_add_zero[i], (y, -1), order='F'))
                coe_reshape_sub_filter_add_zero.append(np.reshape(coe_reshape_add_zero[i], (y, -1), order='F'))
            cur_coll = 1
            # print(coe_reshape_sub_filter_add_zero)
            np_coe_reshape_sub_filter_add_zero = np.array(coe_reshape_sub_filter_add_zero)
            # print(np_coe_reshape_sub_filter_add_zero.shape[2])
            print("np_coe_reshape_sub_filter_add_zero\n", np_coe_reshape_sub_filter_add_zero)
            print("//////////////////////////////////////reduce dim//////////////////////////////////////")
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
    print("======================realignment_data_without_add0 function==========================")
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
    print("polyphase_data_all\n", polyphase_data)
    print("======================Exit realignment_data_without_add0 function==========================")
    return polyphase_data


def realignment_data_with_denominator_z(data, channel_num, D):
    print("======================realignment_data_with_denominator_z function==========================")
    x, y = get_denominator(channel_num, D)
    polyphase_data = np.zeros((x, y, realignment_data_without_add0(data, channel_num, D).shape[1]), dtype=np.complex64)
    print(polyphase_data)
    for i in range(y):
        print("\n", i)
        cur_data_wait_realign = np.concatenate((data[i * M:], np.zeros(i * M)))
        print(cur_data_wait_realign.shape)
        print(cur_data_wait_realign)
        cur_data = realignment_data_without_add0(cur_data_wait_realign, channel_num, D)
        cur_data = np.flipud(cur_data)
        for j in range(cur_data.shape[0]):
            print(cur_data.shape, " ", y - j, " ", y - 1 - i, " ", j)
            polyphase_data[y - j][y - 1 - i] = cur_data[j]
        # print("polyphase_data_all\n", polyphase_data)
    return polyphase_data


def realignment_data_with_z_gcd(data, channel_num, D):
    print("======================realignment_data_with_z_gcd function==========================")
    print("data:\n",data)
    x, y = get_denominator(channel_num, D)
    gcd = math.gcd(channel_num, D)
    print("gcd=", gcd)
    polyphase_data = np.zeros((channel_num, y, realignment_data_without_add0(data, channel_num, D).shape[1]),
                              dtype=np.complex64)
    print("polyphase_data:before for\n",polyphase_data)
    for i in range(y):
        # print("\n", i)
        cur_data_wait_realign = np.concatenate((data[i*gcd:], np.zeros(i * gcd)))
        # cur_data_wait_realign = np.concatenate((data[i*M:], np .zeros(i*M)))
        # print(cur_data_wait_realign.shape)
        print("cur_data_wait_realign:\n",cur_data_wait_realign)
        cur_data = realignment_data_without_add0(cur_data_wait_realign, channel_num, D)
        cur_data = np.flipud(cur_data)
        # print("cur_data after flipud:\n",cur_data)
        for j in range(cur_data.shape[0]):
            # print(cur_data.shape, " ", channel_num - j - 1, " ", y - 1 - i, " ", j)
            polyphase_data[channel_num - j - 1][y - 1 - i] = cur_data[j]
        # print("polyphase_data_all\n", polyphase_data)
    print("polyphase_data:after for\n", polyphase_data)
    return polyphase_data


def polyphase_filter_bank_with_denominator_z(data, filter_coeffs, channel_num, D):
    print("===========================polyphase_filter_bank_with_denominator_z function===============================")
    print(data)
    if M / D == int(M / D):
        print("filter directly ")
        # print(filter_coeffs)
        polyphase_data = realignment_data_without_add0(data, channel_num, D)
        # polyphase_data = np.flipud(polyphase_data)
        # print("polyphase_data after relignment\n", polyphase_data)
        # print("dx polyphase_data after relignment: ", polyphase_data)
        filt_data_conv = []
        print("-------------------------------------------for-------------------------------")
        for i in range(channel_num):
            # print(polyphase_data[i])
            # print(filter_coeffs[i])
            # filt_data_conv.append(np.convolve(polyphase_data[i], filter_coeffs[i],'valid'))
            filt_data_conv.append(np.convolve(polyphase_data[i], filter_coeffs[i]))
            # filt_data_conv.append(scipy.signal.lfilter(filter_coeffs[i], 1, polyphase_data[i]))
        # dispatch_data_conv = scipy.fft.ifft(filt_data_conv, axis=0)
        return filt_data_conv
    else:
        print("filter with sub filter")
        x, y = get_denominator(M, D)
        # polyphase_data = realignment_data_without_add0(data, channel_num, D)   # control data realignment method. *****
        # polyphase_data2 = realignment_data_with_denominator_z(data, channel_num, D)  # control data realignment method. *****
        polyphase_data2 = realignment_data_with_z_gcd(data, channel_num, D)  # control data realignment method. *****
        # polyphase_data = np.flipud(polyphase_data)
        print("----------------------------------after flipud-----------------------------------")
        # print("polyphase_data1\n", polyphase_data2)
        polyphase_data2 = polyphase_data2.reshape(M * y, polyphase_data2.shape[2])
        print("polyphase_data2\n", polyphase_data2)
        # print(filter_coeffs)
        filt_data_conv = []
        final_filt_result = []
        for k in range(channel_num * y):
            # print("\n", k)
            print("cur polyphase_data_sub:\n", polyphase_data2[k])
            print("cur filter_coeffs:\n", filter_coeffs[k])
            filt_data_conv.append(np.convolve(polyphase_data2[k], filter_coeffs[k]))
            # filt_data_conv.append(scipy.signal.lfilter(filter_coeffs[k], 1, polyphase_data2[k]))
            print("filt_data_conv\n", filt_data_conv[k % y])
            print()
            if (k + 1) % y == 0:
                print("add all array for :\n", filt_data_conv)
                final_filt_result.append(sum(filt_data_conv))
                filt_data_conv.clear()
                # print("final filt result of add all conv result")
                # print(final_filt_result)
        # print("\nall final filt result of add all conv result\n", final_filt_result)
        # dispatch_data_conv = scipy.fft.ifft(final_filt_result, axis=0)
        # print("PFB result_data.shape:", final_filt_result.shape)
        return final_filt_result


def consist_all_subband(dx_ospfb_cut, M):
    consist_result = []

    # 计算每个频谱并将其保存到列表中
    for i in range(M):
        freq_spectrum = np.abs(np.fft.fft(dx_ospfb_cut[i], axis=0))
        consist_result.append(freq_spectrum)
        plt.plot(freq_spectrum)
        plt.show()

    # 将所有频谱按顺序拼接在一起
    consist_result_all = np.concatenate(consist_result, axis=0)

    # 绘制拼接后的频谱图
    plt.plot(consist_result_all)
    plt.title('Concatenated Spectrum')
    plt.show()

    return consist_result_all



def compare_pfb(np_data, TAPS, CHANNEL_NUM, D):
    print("----------------------------compare pfb-----------------------------------")
    # print("\nZYZ cspfb result:")
    # zyz_cspfb_coe = cspfb.gen_filter_coeffs(TAPS, CHANNEL_NUM)
    # # print("zyzcoe\n", zyz_cspfb_coe)
    # zyz_cspfb_out = cspfb.polyphase_filter(np_data, zyz_cspfb_coe, CHANNEL_NUM)
    # # print("zyz_cspfb_out\n", zyz_cspfb_out)
    # plot_sub(zyz_cspfb_out, CHANNEL_NUM, "ZYZ cspfb sub band of ")

    # print("\nZYZ ospfb result:")
    # zyz_ospfb_coe = opfb.gen_filter_coeffs(TAPS, CHANNEL_NUM)
    # print("zyzcoe\n", zyz_ospfb_coe)
    # zyz_ospfb_out = opfb.polyphase_filter(np_data, zyz_ospfb_coe, CHANNEL_NUM)
    # print("zyz_ospfb_out\n", zyz_ospfb_out)
    # plot_sub(zyz_ospfb_out, CHANNEL_NUM, "ZYZ ospfb sub band of ")
    #
    print("\nDX " + str(CHANNEL_NUM) + "/" + str(D) + "X ospfb with z gcd result:")
    dx_ospfb_coe = realignment_coe(TAPS, CHANNEL_NUM, D)
    print("dxcoe\n", dx_ospfb_coe)
    dx_ospfb_out = polyphase_filter_bank_with_denominator_z(np_data, dx_ospfb_coe, CHANNEL_NUM, D)
    dx_ospfb_fft = np.fft.ifft(dx_ospfb_out, axis=0)
    print(dx_ospfb_fft.shape)
    plot_sub(dx_ospfb_fft, CHANNEL_NUM, "DX " + str(CHANNEL_NUM) + "/" + str(D) + "X ospfb with z gcd result:")
    #
    print("\nDX " + str(CHANNEL_NUM) + "/" + str(D) + "X ospfb with z gcd and rotate result:")
    dx_ospfb_coe = realignment_coe(TAPS, CHANNEL_NUM, D)
    dx_ospfb_out = polyphase_filter_bank_with_denominator_z(np_data, dx_ospfb_coe, CHANNEL_NUM, D)
    dx_ospfb_rotate = circular_rotate(dx_ospfb_out, CHANNEL_NUM, D)
    dx_ospfb_fft = np.fft.ifft(dx_ospfb_rotate, axis=0)
    plot_sub(dx_ospfb_fft, CHANNEL_NUM,
             "DX " + str(CHANNEL_NUM) + "/" + str(D) + "X ospfb with z gcd and rotate result:")
    #
    print("\nDX " + str(CHANNEL_NUM) + "/" + str(D) + "X ospfb with z gcd rotate and cut result:")
    dx_ospfb_coe = realignment_coe(TAPS, CHANNEL_NUM, D)
    dx_ospfb_out = polyphase_filter_bank_with_denominator_z(np_data, dx_ospfb_coe, CHANNEL_NUM, D)
    dx_ospfb_rotate = circular_rotate(dx_ospfb_out, CHANNEL_NUM, D)
    dx_ospfb_cut = cut_extra_channel_data_by_tail(np.fft.fft(np.fft.ifft(dx_ospfb_rotate, axis=0)), CHANNEL_NUM,
                                                  D) * D / M
    plot_sub(np.fft.ifft(dx_ospfb_cut), CHANNEL_NUM,
             "DX " + str(CHANNEL_NUM) + "/" + str(D) + "X ospfb with z gcd and rotate cut result:")

    print("\nDX " + str(CHANNEL_NUM) + "/" + str(D) + "X ospfb with z gcd rotate cut and consist result:")
    dx_ospfb_coe = realignment_coe(TAPS, CHANNEL_NUM, D)
    dx_ospfb_out = polyphase_filter_bank_with_denominator_z(np_data, dx_ospfb_coe, CHANNEL_NUM, D)
    dx_ospfb_rotate = circular_rotate(dx_ospfb_out, CHANNEL_NUM, D)
    dx_ospfb_cut = cut_extra_channel_data_by_tail(np.fft.fft(np.fft.ifft(dx_ospfb_rotate, axis=0)), CHANNEL_NUM,
                                                  D) * D / M
    print("dx_ospfb_cut:\n",dx_ospfb_cut)
    dx_ospfb_consist = consist_all_subband(np.fft.ifft(dx_ospfb_cut),M)

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

        print("subpfb")
        flag = True

        blocks = pow(2, 25)
        filename = "./PFB-main/PFB-main/data/J0437-4715.dada"

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
                    # np.savetxt("txt/Original_pol1_block0_data.txt", plot_data_pol1)
                    # np.savetxt("txt/Original_pol2_block0_data.txt", plot_data_pol2)
                    plt.legend(prop={'size': 14})
                    # plt.yticks([0.0, 1.1e7, 2.2e7], [0, 0.5, 1], size=14)
                    # plt.xticks([0, 16384], [1582, 1182], size=14)
                    plt.xlabel('Frequency (MHz)', size=14)
                    plt.ylabel('Normalized Amplitude', size=14)
                    # plt.savefig("img/imag_after_cutff/original_2channel.jpg",dpi=400)
                    plt.show()

                    plt.figure(figsize=(15, 7))
                    plot_data_pol1 = np.abs(result_consist_pol1)[
                             :int(result_consist_pol1.size / 2)]
                    plot_data_pol2=np.abs(result_consist_pol2)[
                             :int(result_consist_pol2.size / 2)]
                    plt.plot(plot_data_pol2, label="2x IOSC Pol 2")
                    plt.plot(plot_data_pol1, label="2x IOSC Pol 1")
                    np.savetxt("txt/Ospfb2x_pol1_block0_data_8tap.txt", plot_data_pol1)
                    np.savetxt("txt/Ospfb2x_pol2_block0_data_8tap.txt", plot_data_pol2)
                    plt.legend(prop={'size': 14})
                    plt.yticks([0.0, 0.65e6, 1.3e6], [0, 0.5, 1], size=14)
                    plt.xticks([0, 16384], [1582, 1182], size=14)
                    plt.xlabel('Frequency (MHz)', size=14)
                    plt.ylabel('Normalized Amplitude', size=14)
                    plt.savefig("img/imag_after_cutff/1024/ospfb2x_4channel_8tap.jpg", dpi=400)
                    plt.show()

                    os._exit()

                print("PFB done, use ", time.time() - cur_spend_time, " seconds, all spend ",
                      time.time() - start_time, " seco nds")
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
                    # np.savetxt("txt/Original_pol1_block0_data.txt", plot_data_pol1)
                    # np.savetxt("txt/Original_pol2_block0_data.txt", plot_data_pol2)
                    plt.legend(prop={'size': 14})
                    plt.yticks([0.0, 1.0e7, 2.0e7], [0, 0.5, 1], size=14)
                    plt.xticks([0, 16384], [1582, 1182], size=14)
                    plt.xlabel('Frequency (MHz)', size=14)
                    plt.ylabel('Normalized Amplitude', size=14)
                    # plt.savefig("img/imag_after_cutff/original_2channel.jpg",dpi=400)
                    plt.show()

                    # draw Critically sampled channelizer figure
                    plt.figure(figsize=(15, 7))
                    plot_data_pol1=np.abs(result_consist_pol1)[
                             :int(result_consist_pol1.size / 2)]
                    plot_data_pol2=np.abs(result_consist_pol2)[
                             :int(result_consist_pol2.size / 2)]
                    plt.plot(plot_data_pol2, label="CSC Pol 2")
                    plt.plot(plot_data_pol1, label="CSC Pol 1")
                    np.savetxt("txt/Cspfb_pol1_block0_data_8tap.txt", plot_data_pol1)
                    np.savetxt("txt/Cspfb_pol2_block0_data_8tap.txt", plot_data_pol2)
                    plt.yticks([0.0, 0.65e6, 1.3e6], [0, 0.5, 1],size=14)
                    plt.xticks([0, 16384], [1582, 1182],size=14)
                    plt.xlabel('Frequency (MHz)', size=14)
                    plt.ylabel('Normalized Amplitude', size=14)
                    plt.legend(prop={'size': 14})
                    plt.savefig("img/imag_after_cutff/1024/cspfb_4channel_8tap.jpg",dpi=400)
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
            np.savetxt("txt/ospfb.txt", idata)
        else:
            np.savetxt("txt/cspfb.txt", idata)

        plt.figure(figsize=(10, 5), dpi=100)
        # plt.title("Final")
        plt.ylabel("Magnitude(dB)")
        plt.xlabel("Phase")
        plt.plot(np.abs(idata))
        plt.show()

    subpfb()


if __name__ == '__main__':
    conv()

    TAPS = 63
    CHANNEL_NUM = 4
    M = CHANNEL_NUM
    D = 3
    print("----------------------------------start---------------------------------------")
    reset_save_data()

    # 1.测试PFB（支持临界采样、整数倍过采样、分数倍过采样，只需修改通道数CHANNEL_NUM和D抽取因子即可，TAPS为滤波器抽头数）:
    data = []
    for i in range(36):
        data.append(i)
    np_data = np.array(data)
    print(np_data)
    # circular_rotate(np_data, CHANNEL_NUM,D)
    # ////////////////////////////////////////////////////////////////////////
    np_data = np.loadtxt(r'PFB-main\PFB\mini_data.txt')
    # plt.plot(np.abs(np.fft.fft(np_data)))
    # plt.show()
    np_data = add_rfi2(np_data, CHANNEL_NUM)
    plt.figure(figsize=(6, 5), dpi=400)
    # plt.xticks([0, 6400,12800], [-1,0,1], size=14)
    plt.yticks([0, 40000,80000], [0,0.5,1], size=14)
    plt.xlabel('Sampling Points Number', size=14)
    plt.ylabel('Normalized Amplitude', size=14)
    plt.plot(np.abs(np.fft.fft(np_data)))
    plt.savefig("img/imag_after_cutff/1117/before_pfb.jpg",dpi=400)
    plt.show()
    compare_pfb(np_data, TAPS, CHANNEL_NUM, D)
    # ////////////////////////////////////////////////////////////////////////

    # 2.PFB并消色散，保存数据到profile中，需关闭测试代码
    # coherent_dedispersion(TAPS, CHANNEL_NUM, D)

    # 3.消色散后profile由当前目录的jupyter文件绘图，注意修改jupyter代码中的profile文件名（subospfb.txt/subcspfb.txt）
    # plot_profile.ipynb
    # plot_sub.ipynb

    print("--------------------------------------------end------------------------------------------------------")
