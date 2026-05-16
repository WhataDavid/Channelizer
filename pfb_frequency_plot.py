import numpy as np
import math
import scipy
from scipy.signal import chirp, remez, firwin, freqz
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

"""
生成系数
"""
def generate_win_coeffs(numtaps, M, radio = 1, window_fn="hamming"):    
    #抽头数、分支数、窗函数类型
    win_coeffs = scipy.signal.get_window(window_fn, numtaps*M)   
    
    #返回给定类型和长度的窗，窗口的样本数量M*P，这是一个数组
    sinc       = scipy.signal.firwin(numtaps*M, cutoff=radio / M, window="hamming")   
    #sinc函数                                                                              
    #采用窗函数法设计FIR滤波器，这个函数计算FIR滤波器的系数。
    #滤波器的长度（系数的数目）为M*P；滤波器的截止频率  
    win_coeffs *= sinc   #样本数量和FIR滤波器系数相乘
    win_coeffs = np.reshape(win_coeffs, (M, -1), order='F')
    return win_coeffs

def pfb(data,M,win_coeffs):
    # 使数据长度为M的整数倍，不够整数倍的部分则补零
    disp_len = int(np.ceil(data.size / M))
    patch_size = int(disp_len * M - data.size)
    patch_data = np.concatenate((data, np.zeros(patch_size)))

    # 将数据重新排序为 M行XX列的矩阵   
    reshape_data = np.reshape(patch_data, (M, -1), order='F')
    # 将数据上下翻转
    polyphase_data = np.flipud(reshape_data)

    nv = np.arange(disp_len)
    prefilt_data = polyphase_data * ((-1) ** nv)

    # Polyphase filter bank
    filt_data = np.zeros(prefilt_data.shape, dtype=complex)
    for k in range(M):
        filt_data[k] = scipy.signal.lfilter(win_coeffs[k],1,prefilt_data[k])
        # filt_data[k] = scipy.signal.filtfilt(win_coeffs[k],1,prefilt_data[k])

    # 转换为复数
    postfilt_data = np.zeros(prefilt_data.shape, dtype=complex)
    for k in range(M):
        postfilt_data[k] = filt_data[k] * ((-1) ** k) * np.exp(-1j * np.pi * k / M)
        
    # FFT
    dispatch_data = scipy.fft.fft(postfilt_data, axis=0)

    return dispatch_data

def sweep_freqz(M,N,win_coeffs): 
    data = np.zeros(N)
    w = np.linspace(0, 1, N)
    p = 2 * np.pi * np.cumsum(w)
    data = np.exp(1j * p)   
    pfb_data = pfb(data,M,win_coeffs)
    a = 20 * np.log10(np.abs(pfb_data))
    return a

def plot(M, w, a):
    plt.figure(figsize=(10, 6))
    for i in range(M):
        plt.plot(w,a[i].T, linewidth=4,label=f'Channel {i+1}')

    # plt.title('8-Channel Polyphase Filter Bank Frequency Response with 4/3 Oversampling')
    plt.title('8-Channel Polyphase Filter Bank Frequency Response')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude [dB]')
    #plt.legend()
    plt.grid()
    plt.savefig("PFB_sub-band.svg",dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    M = 8    # 通道数、分支数
    N = 12800
    numtaps = 64    
    fs = 800e6
    freq = 400
    radio = 4 / 3
    # radio = 1
    win_coeffs = generate_win_coeffs(numtaps, M, radio)
    a = sweep_freqz(M,N, win_coeffs)
    w = np.linspace(0, freq, N // M )

    plot(M, w, a)

