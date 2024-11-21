import numpy as np
import matplotlib.pyplot as plt
import math


def fft_basic_2(x, N=4):
    # 频域抽取的基2FFT
    loop_num = int(np.log2(N))
    data = np.zeros((loop_num + 1, N), dtype=np.complex128)
    data[0] = x

    for i in range(loop_num):
        k = i + 1
        for p in range(2 ** i):
            for j in range(N // (2 ** k)):
                data[i + 1][j + p * (N // (2 ** i))] = data[i, j + p * (N // (2 ** i))] + data[
                    i, j + N // (2 ** k) + p * (N // (2 ** i))]
                data[i + 1][j + N // (2 ** k) + p * (N // (2 ** i))] = (data[i, j + p * (N // (2 ** i))] - data[
                    i, j + N // (2 ** k) + p * (N // (2 ** i))]) * np.e ** (-1j * 2 * j * np.pi * (2 ** i) / N)

    def rev2(k, N):
        if (k == 0):
            return (0)
        else:
            return (((rev2(k // 2, N) // 2) + (k % 2) * (N // 2)))

    # 输出倒序
    fft_out = np.ones_like(data[0, :])
    for k in range(N):
        fft_out[rev2(k, N)] = data[loop_num, k]

    return fft_out


def fft_basic_5(x, N=5):
    # 频域抽取的基5FFT
    loop_num = int(round(math.log(N, 5)))
    data = np.zeros((loop_num + 1, N), dtype=np.complex64)
    data[0, :] = x

    for i in range(loop_num):
        k = i + 1
        for p in range(5 ** i):
            for j in range(N // (5 ** k)):
                data[i + 1][j + 5 * p * (N // (5 ** k))] = data[i, j + 5 * p * (N // (5 ** k))] \
                                                           + data[i, j + N // (5 ** k) + 5 * p * (N // (5 ** k))] \
                                                           + data[i, j + 2 * N // (5 ** k) + 5 * p * (N // (5 ** k))] \
                                                           + data[i, j + 3 * N // (5 ** k) + 5 * p * (N // (5 ** k))] \
                                                           + data[i, j + 4 * N // (5 ** k) + 5 * p * (N // (5 ** k))]
                data[i + 1][j + N // (5 ** k) + 5 * p * (N // (5 ** k))] = (data[i, j + 5 * p * (N // (5 ** k))] \
                                                                            + data[i, j + N // (5 ** k) + 5 * p * (
                                N // (5 ** k))] * np.e ** (-1j * 1 * 2 * np.pi / 5) \
                                                                            + data[i, j + 2 * N // (5 ** k) + 5 * p * (
                                N // (5 ** k))] * np.e ** (-1j * 2 * 2 * np.pi / 5) \
                                                                            + data[i, j + 3 * N // (5 ** k) + 5 * p * (
                                N // (5 ** k))] * np.e ** (-1j * 3 * 2 * np.pi / 5)
                                                                            + data[i, j + 4 * N // (5 ** k) + 5 * p * (
                                N // (5 ** k))] * np.e ** (-1j * 4 * 2 * np.pi / 5)) * np.e ** (
                                                                                   -1j * 2 * j * np.pi * (
                                                                                   5 ** i) / N)
                data[i + 1][j + 2 * N // (5 ** k) + 5 * p * (N // (5 ** k))] = (data[i, j + 5 * p * (N // (5 ** k))]
                                                                                + data[i, j + N // (5 ** k) + 5 * p * (
                                N // (5 ** k))] * np.e ** (-1j * 2 * 2 * np.pi / 5)
                                                                                + data[
                                                                                    i, j + 2 * N // (5 ** k) + 5 * p * (
                                                                                            N // (
                                                                                            5 ** k))] * np.e ** (
                                                                                        -1j * 4 * 2 * np.pi / 5)
                                                                                + data[
                                                                                    i, j + 3 * N // (5 ** k) + 5 * p * (
                                                                                            N // (
                                                                                            5 ** k))] * np.e ** (
                                                                                        -1j * 1 * 2 * np.pi / 5)
                                                                                + data[
                                                                                    i, j + 4 * N // (5 ** k) + 5 * p * (
                                                                                            N // (
                                                                                            5 ** k))] * np.e ** (
                                                                                        -1j * 3 * 2 * np.pi / 5)) * np.e ** (
                                                                                       -1j * 4 * j * np.pi * (
                                                                                       5 ** i) / N)
                data[i + 1][j + 3 * N // (5 ** k) + 5 * p * (N // (5 ** k))] = (data[i, j + 5 * p * (N // (5 ** k))]
                                                                                + data[i, j + N // (5 ** k) + 5 * p * (
                                N // (5 ** k))] * np.e ** (-1j * 3 * 2 * np.pi / 5)
                                                                                + data[
                                                                                    i, j + 2 * N // (5 ** k) + 5 * p * (
                                                                                            N // (
                                                                                            5 ** k))] * np.e ** (
                                                                                        -1j * 1 * 2 * np.pi / 5)
                                                                                + data[
                                                                                    i, j + 3 * N // (5 ** k) + 5 * p * (
                                                                                            N // (
                                                                                            5 ** k))] * np.e ** (
                                                                                        -1j * 4 * 2 * np.pi / 5)
                                                                                + data[
                                                                                    i, j + 4 * N // (5 ** k) + 5 * p * (
                                                                                            N // (
                                                                                            5 ** k))] * np.e ** (
                                                                                        -1j * 2 * 2 * np.pi / 5)) * np.e ** (
                                                                                       -1j * 6 * j * np.pi * (
                                                                                       5 ** i) / N)
                data[i + 1][j + 4 * N // (5 ** k) + 5 * p * (N // (5 ** k))] = (data[i, j + 5 * p * (N // (5 ** k))]
                                                                                + data[i, j + N // (5 ** k) + 5 * p * (
                                N // (5 ** k))] * np.e ** (-1j * 4 * 2 * np.pi / 5)
                                                                                + data[
                                                                                    i, j + 2 * N // (5 ** k) + 5 * p * (
                                                                                            N // (
                                                                                            5 ** k))] * np.e ** (
                                                                                        -1j * 3 * 2 * np.pi / 5)
                                                                                + data[
                                                                                    i, j + 3 * N // (5 ** k) + 5 * p * (
                                                                                            N // (
                                                                                            5 ** k))] * np.e ** (
                                                                                        -1j * 2 * 2 * np.pi / 5)
                                                                                + data[
                                                                                    i, j + 4 * N // (5 ** k) + 5 * p * (
                                                                                            N // (
                                                                                            5 ** k))] * np.e ** (
                                                                                        -1j * 1 * 2 * np.pi / 5)) * np.e ** (
                                                                                       -1j * 8 * j * np.pi * (
                                                                                       5 ** i) / N)

    # 递推计算倒序，FPGA中可使用五进制进行倒序
    def rev5(k, N):
        if (k == 0):
            return (0)
        else:
            return (((rev5(k // 5, N) // 5) + (k % 5) * (N // 5)))

    # 输出倒序
    fft_out = np.ones_like(data[0, :])
    for k in range(N):
        fft_out[rev5(k, N)] = data[loop_num, k]
    return fft_out


def fft_recursion(x):
    N = len(x)
    if N <= 1: return x
    even = fft_recursion(x[0::2])
    odd = fft_recursion(x[1::2])
    T = [np.exp(-2j * np.pi * k / N) * odd[k] for k in range(N // 2)]
    return [even[k] + T[k] for k in range(N // 2)] + \
        [even[k] - T[k] for k in range(N // 2)]


def cus_fft_20(x):
    print(x)
    input_array = pre_reorder(x)
    result_first = np.zeros(shape=(4, 5), dtype=np.complex128)
    print(result_first)
    for i in range(4):
        # result_first[i] = fft_basic_5(input_array[i])
        result_first[i] = winograd_fft(input_array[i])
        # result_first[i] = np.fft.fft(input_array[i])
    print(result_first)
    middle_reorder_array = middle_reorder(result_first)
    result_second = np.zeros(shape=(5, 4), dtype=np.complex128)
    for i in range(5):
        result_second[i] = fft_basic_2(middle_reorder_array[i])
        # result_second[i] = np.fft.fft(middle_reorder_array[i])

    print("11111111111111111111")
    print(result_second[0][0])
    print(result_second[1][1])
    print(result_second[2][2])
    print(result_second[3][3])
    print(result_second[4][0])
    print(result_second[0][1])
    print(result_second[1][2])
    print(result_second[2][3])
    print(result_second[3][0])
    print(result_second[4][1])
    print(result_second[0][2])
    print(result_second[1][3])
    print(result_second[2][0])
    print(result_second[3][1])
    print(result_second[4][2])
    print(result_second[0][3])
    print(result_second[1][0])
    print(result_second[2][1])
    print(result_second[3][2])
    print(result_second[4][3])
    print('--------------------')
    result_final = final_reorder(result_second)
    np_result = np.fft.fft(x)
    for i in range(len(np_result)):
        print("result_final[", i, "]\n", result_final[i])
        print("np_result[", i, "]\n", np_result[i])
        print(abs(result_final[i] - np_result[i]))
        print()


def pre_reorder(x):
    reshapeX = np.reshape(x, (4, 5), order='F')
    print(reshapeX)
    for k in range(4):
        reshapeX[k] = np.roll(reshapeX[k], -k)
    print(reshapeX)
    return reshapeX


def middle_reorder(x):
    reshapeX = np.transpose(x)
    print(reshapeX)
    return reshapeX


def final_reorder(x):
    print("final_reorder_function\n", x)
    for k in range(5):
        x[k] = np.roll(x[k], -k)
        print(x[k])
    x = x.transpose()
    x = np.ravel(x)
    print(x)
    print("end_final_reorder_function\n")
    return x


def matrix_dft():
    A = np.arange(0, 25).reshape(5, 5)
    B = np.ones((5, 1))
    print(A)
    print('------------')
    print(B)
    print('------------')
    print(np.dot(A, B))
    W = np.e ** (-1j * 2 * np.pi / 5)
    print(W)
    Alist = [1, 1, 1, 1, 1, 1, W, W ** 2, W ** 3, W ** 4, 1, W ** 2, W ** 4, W, W ** 3, 1, W ** 3, W, W ** 4, W ** 2, 1,
             W ** 4, W ** 3, W ** 2, W]
    A = np.array(Alist).reshape(5, 5)
    print(A)
    Blist = [0, 1, 2, 3, 4]
    B = np.array(Blist).reshape(5, 1)
    print(B)
    C = np.dot(A, B)
    print(C)
    print(np.fft.fft([0, 1, 2, 3, 4]))


def winograd_fft(x):
    W = np.e ** (-1j * 2 * np.pi / 5)
    # print(W)
    Alist = [W, W ** 3, W ** 4, W ** 2, W ** 2, W ** 1, W ** 3, W ** 4, W ** 4, W ** 2, W, W ** 3, W ** 3, W ** 4,
             W ** 2, W]
    A = np.array(Alist).reshape(4, 4)
    # print(A)
    Blist = [x[1], x[3], x[4], x[2]]
    B = np.array(Blist).reshape(4, 1)
    # print(B)
    C = np.dot(A, B)
    # print(C)
    y0 = sum(x)
    y1 = C[0][0] + x[0]
    y2 = C[1][0] + x[0]
    y3 = C[2][0] + x[0]
    y4 = C[3][0] + x[0]
    y=[y0,y1,y2,y4,y3]
    print(y)
    # print(np.fft.fft(x))
    y_array = np.array(y)
    # print("y_array",y_array)
    return y_array
# N=4
# X=[1,2,3,4]
# print("np_fft_out", np.fft.fft(X))
# print("cus_fft_out", fft_basic_2(N,X))
# print("fft_recursion", fft_recursion(X))
# print()
# N=5
# X=[1,2,3,4,5]
# print("np_fft_out", np.fft.fft(X))
# print("cus_fft_out", fft_basic_5(N,X))
#
# print()
# N=8
# X=[1,2,3,4,5,6,7,8]
# print("np_fft_out", np.fft.fft(X))
# print("fft_recursion", fft_recursion(X))

# print(fft_basic_5([0, 4, 8, 12, 16]))
# print(fft_basic_5([5, 9, 13, 17, 1]))
# print(fft_basic_5([10, 14, 18, 2, 6]))
# print(fft_basic_5([15, 19, 3, 7, 11]))
X = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
X2 = X[::-1]
print(X)
print(X2)
# cus_fft_20(X)
print(np.fft.fft(X))
print(np.fft.ifft(X2))


# winograd_fft([0, 1, 2, 3, 4])
