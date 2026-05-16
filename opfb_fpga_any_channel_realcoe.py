import numpy as np
import scipy.signal
import matplotlib.pyplot as plt


def plot_sub(data, M, title):
    plt.figure(figsize=(16, 12))
    plt.suptitle(title, size=20, fontweight='bold', y=0.98)

    for i in range(M):
        plt.subplot(4, 4, i + 1)

        # 为了画图极其清晰，给每个通道的数据加一个 Blackman 窗消除频谱泄漏
        window = scipy.signal.windows.blackman(len(data[i]))
        fft_data = np.abs(np.fft.fftshift(np.fft.fft(data[i] * window)))

        # 归一化方便观察
        if np.max(fft_data) > 1e-5:
            fft_data = fft_data / np.max(fft_data)

        plt.plot(fft_data, color='b')

        # 判断如果是全零信号，坐标轴固定
        plt.ylim(0, 1.1)
        plt.title(f"Channel {i}", size=14, fontweight='bold')

        # 隐藏X轴刻度保持清爽
        plt.xticks([])
        plt.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def filtercoes(taps, M, D):
    N = M * taps
    # 纯实数低通滤波器
    coe = scipy.signal.firwin(N, cutoff=1.0 / D, window=("kaiser", 6))
    return np.array(coe, dtype=np.float64)


def cir_data(data, M, D):
    if M == D:
        return data
    arr_data = data.T
    step = 0
    for i in range(arr_data.shape[0]):
        if step != 0:
            arr_data[i] = np.roll(arr_data[i], step)
        step = (step + (M - D)) % M
    return arr_data.T


def opfb(data, coe, taps, M, D):
    data = np.array(data, dtype=np.float64)
    data_snake_array = np.zeros(M * taps, dtype=np.float64)
    alldata = []

    while len(data) >= D:
        # 向右移位 D 个，开头填入新的 D 个（逆序）
        data_snake_array[D:] = data_snake_array[:-D]
        data_snake_array[:D] = data[:D][::-1]
        data = data[D:]

        res = np.zeros(M, dtype=np.float64)
        for i in range(0, M * taps):
            res[i % M] += np.round(data_snake_array[i] * coe[i], 7)
        alldata.append(res)

    flip_data = np.array(alldata).T
    outdata_cir = cir_data(flip_data, M, D)

    # 1. 执行 IFFT 得到多相输出
    dx_ospfb_fft = np.fft.ifft(outdata_cir, axis=0)

    # ==============================================================
    # 2. 极其关键：OSPFB 基带相位旋转补偿 (消除 Nyquist 折叠)
    # 把因为 D != M 造成的各通道频移完美修正到 0 Hz 中间
    # ==============================================================
    num_out_samples = dx_ospfb_fft.shape[1]
    m_idx = np.arange(num_out_samples)
    for k in range(M):
        # 修正因子： e^(-j * 2 * pi * m * k * D / M)
        correction = np.exp(-1j * 2 * np.pi * k * D / M * m_idx)
        dx_ospfb_fft[k] *= correction

    return dx_ospfb_fft


def generate_exact_tones(fs, M, Ns):
    print(f"Generating Precise Tone Test Data ({Ns} samples)...")
    t = np.arange(Ns) / fs
    data = np.zeros(Ns, dtype=np.float64)

    # 定义子带频率间隔，确保信号落在FFT的正中心，不产生一点毛刺
    df = fs / Ns

    def add_tone(ch, offset_bins):
        # ch 是通道号， offset_bins 是在通道中心左右偏移的频点数
        f = ch * (fs / M) + offset_bins * df
        return np.cos(2 * np.pi * f * t)

    # 巧妙设计：
    # Ch 1: 注入 1 根天线
    data += add_tone(1, 0)

    # Ch 2: 注入 2 根天线 (左右各偏移 100 个频点)
    data += add_tone(2, -100)
    data += add_tone(2, 100)

    # Ch 3: 注入 3 根天线
    data += add_tone(3, -200)
    data += add_tone(3, 0)
    data += add_tone(3, 200)

    # Ch 4: 注入 4 根天线
    data += add_tone(4, -300)
    data += add_tone(4, -100)
    data += add_tone(4, 100)
    data += add_tone(4, 300)

    # 其余通道全是 0！
    return data


if __name__ == "__main__":
    TAPS = 128
    M = 16
    D = 12
    fs = 2.064e9

    # 选取 196608 (能被 12 整除，且刚好是 16384 的整数倍，FFT最好算)
    Ns = 196608

    # 1. 生成纯实数精确单音信号
    data = generate_exact_tones(fs, M, Ns)

    # (仅作参考) 画一下原始输入信号的宽带频谱
    plt.figure(figsize=(12, 4))
    plt.plot(np.abs(np.fft.fft(data)[:Ns // 2]), 'm')
    plt.title("Original Input Spectrum (Only shows signals in Ch1, Ch2, Ch3, Ch4)")
    plt.grid(True)
    plt.show()

    # 2. 生成实数滤波系数
    coe = filtercoes(TAPS, M, D)

    # 3. 运行 OPFB
    print(f"Running OPFB (M={M}, D={D})...")
    out_data = opfb(data, coe, TAPS, M, D)

    # 4. 绘图验证
    print("Plotting results...")
    plot_sub(out_data, M, title=f"OPFB Precise Test (M={M}, D={D})")