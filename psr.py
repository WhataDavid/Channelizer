import numpy as np
import scipy
import time


def apply_chirp(freq, size, bw=400.0, fc=1382.0):
    dm = 2.64476
    dm_dispersion = 2.41e-4
    dispersion_per_MHz = 1e6 * dm / dm_dispersion
    phasors = np.zeros((size),dtype=complex)
    binwidth = -bw / size
    coeff = 2 * np.pi * dispersion_per_MHz / (fc * fc);
    for i in range (size):
        f = i * binwidth + 0.5 * bw
        phasors[i] = np.exp(1j * coeff * f * f / (fc+f));
        if i == 0:
            phasors[i] = 0;
        freq[i] = phasors[i] * freq[i];


def apply_chirp_optimize(freq, size, bw=400.0, fc=1382.0):
    start_time = time.time()
    dm = 2.64476
    dm_dispersion = 2.41e-4
    dispersion_per_MHz = 1e6 * dm / dm_dispersion
    binwidth = -bw / size
    coeff = 2 * np.pi * dispersion_per_MHz / (fc * fc)

    # 生成频率数组
    f = np.linspace(0.5 * bw, -0.5 * bw, size)

    # 计算相位因子
    phasors = np.exp(1j * coeff * f * f / (fc + f))
    phasors[0] = 0  # 保持第一个元素为0

    # 应用相位因子
    freq *= phasors
    print("apply_chirp_optimize use time: ", time.time() - start_time)


def get_period_size(bw):
    start_time = time.time()
    period = 0.00575730363767324  # 400M
    # period = 0.0006919836102972644 # 3328M
    # period = 0.00575730363767324  # 10M
    print("get_period_size use time: ", time.time() - start_time)
    return int(period * bw * 1e6)


def fold_integral(data, bw, first_phase, hits, profile):
    start_time = time.time()
    nbin = 1024.0
    pfold = 0.0006919836102972644
    sampling_interval = 1 / (bw * 1e6)
    phase_per_sample = sampling_interval / pfold
    phi = np.zeros(data.shape, dtype=int)
    # profile = np.zeros((int)(nbin))
    # hits = np.zeros((int)(nbin))
    iphi = 0.0
    for i in range(data.shape[0]):
        iphi = first_phase + i * phase_per_sample
        iphi -= np.floor(iphi)
        myphi = int(iphi * nbin)
        phi[i] = myphi
    for i in range(data.shape[0]):
        index = int(phi[i])
        hits[index] += 1
        profile[index] += data[i]

    first_phase = iphi
    print("first_phase_shape", first_phase.shape)
    print("first_phase\n", first_phase)
    print("fold_integral use time: ", time.time() - start_time)
    return first_phase


def fold_integral_optimize(data, bw, first_phase, hits, profile):
    start_time = time.time()
    nbin = 1024.0
    pfold = 0.0006919836102972644
    sampling_interval = 1 / (bw * 1e6)
    phase_per_sample = sampling_interval / pfold

    # 计算相位
    iphi = first_phase + np.arange(data.shape[0]) * phase_per_sample
    iphi -= np.floor(iphi)
    phi = (iphi * nbin).astype(int)

    # 使用 NumPy 的 bincount 函数进行累加
    np.add.at(hits, phi, 1)
    np.add.at(profile, phi, data)

    first_phase = iphi[-1]
    # print("first_phase_shape", first_phase.shape)
    # print("first_phase\n",first_phase)
    print("fold_integral use time: ", time.time() - start_time)
    return first_phase


def fold_data(data, blocks, psize, pdata, pnum, location):
    start_time = time.time()
    cur = location
    # pdata = np.zeros((psize))
    # pnum = np.zeros((psize))
    for i in range(blocks):
        if (cur >= psize):
            cur = 0
        pnum[cur] = pnum[cur] + 1
        pdata[cur] = (pnum[cur] - 1) * pdata[cur] / pnum[cur] + data[i] / pnum[cur]
        cur = cur + 1
    location = cur
    print("fold_data use time: ", time.time() - start_time)
    return location


def fold_data_optimize(data, blocks, psize, pdata, pnum, location):
    start_time = time.time()
    cur = location

    for i in range(blocks):
        if cur >= psize:
            cur = 0
        pnum_cur = pnum[cur]
        pnum[cur] += 1
        pdata[cur] = (pnum_cur * pdata[cur] + data[i]) / pnum[cur]
        cur += 1

    location = cur
    print("fold_data_optimize use time: ", time.time() - start_time)
    return location


# 示例调用
blocks = 1024
psize = 512
data = np.random.random(blocks)
pdata = np.zeros(psize)
pnum = np.zeros(psize)
location = 0

location = fold_data(data, blocks, psize, pdata, pnum, location)


def integral_data1(data, size, n=1024):
    start_time = time.time()
    data = np.abs(data)
    bins = size // n
    out = np.zeros((1024))
    sum = 0
    j = 0
    z = 0
    for i in range(bins * n):
        if j == bins:
            out[z] = sum / bins
            z += 1
            sum = 0
            j = 0
        j += 1
        sum += data[i]
    print("integral_data1 use time: ", time.time() - start_time)
    return out


def integral_data(data, size, n=1024):
    start_time = time.time()
    data = np.abs(data)
    bins = size // n
    out = np.zeros((1024))
    sum = 0
    index = 0
    i = 0
    while (1):
        sum = 0
        j = 0
        while (1):
            if j > bins or i > (bins * (n - 1)):
                break
            sum += data[i]
            i += 1
            j += 1
        out[index] = sum / j
        index += 1
        # print(i)
        if i > (bins * (n - 1)):
            break
    sum = 0
    while (1):
        if i >= size:
            break
        sum += data[i]
        i += 1

    out[n - 1] = sum / (size - n * bins + bins)
    print("integral_data use time: ", time.time() - start_time)
    return out


def integral_data_opfb(data, psize, nchannels):
    start_time = time.time()
    num = nchannels // 2 + 1
    subpsize = psize // (num - 1)
    idata = np.zeros((1024 * num))
    start = 0
    end = 0
    size = 0
    for i in range(num):
        start += size
        if i == 0 or i == (num - 1):
            size = subpsize // 2
        else:
            size = subpsize
        end = start + size
        idata[i * 1024:(i + 1) * 1024] = integral_data(data[start:end], size)
    print("integral_data_opfb use time: ", time.time() - start_time)
    return idata


def integral_data_cspfb(data, psize, nchannels):
    start_time = time.time()
    num = nchannels // 2
    subpsize = psize // num
    idata = np.zeros((1024 * num))
    start = 0
    end = 0
    size = 0
    for i in range(num):
        start += size
        size = subpsize
        end = start + size
        idata[i * 1024:(i + 1) * 1024] = integral_data(data[start:end], size)
    print("integral_data_cspfb use time: ", time.time() - start_time)
    return idata


def coherent_dedispersion(pol1, pol2, num):
    start_time = time.time()
    num = num // 2
    pol1_f = scipy.fft.fft(pol1)[:num]
    pol2_f = scipy.fft.fft(pol2)[:num]
    apply_chirp(pol1_f, num)
    apply_chirp(pol2_f, num)
    pol1_t = scipy.fft.ifft(pol1_f)
    pol2_t = scipy.fft.ifft(pol2_f)
    pol_out = np.sqrt(np.abs(pol1_t) ** 2 + np.abs(pol2_t) ** 2)
    print("coherent_dedispersion use time: ", time.time() - start_time)
    return pol_out


def coherent_dedispersion_pfb(pol1, pol2, num):
    start_time = time.time()
    num = num // 2
    apply_chirp(pol1, num)
    apply_chirp(pol2, num)
    pol1_t = scipy.fft.ifft(pol1)
    pol2_t = scipy.fft.ifft(pol2)
    pol_out = np.sqrt(np.abs(pol1_t) ** 2 + np.abs(pol2_t) ** 2)
    print("coherent_dedispersion_pfb use time: ", time.time() - start_time)
    return pol_out


def coherent_dedispersion_opfb(pol1, pol2, nchannels, pdata, pnum, location):
    start_time = time.time()
    freq_size = pol1.shape[1]
    num = nchannels // 2 + 1
    start = 0
    end = 0
    bandwidth = 10 / (num - 1)
    bw = 0
    size = 0
    fc = 1267
    fstart = 0
    fend = 0
    psize = 0
    for i in range(num):
        fc -= bw / 2
        if i == 0:
            start = freq_size // 2
            end = freq_size // 2 + freq_size // 4
            bw = bandwidth / 2
        elif i == 8:
            start = freq_size // 2 - freq_size // 4
            end = freq_size // 2
            bw = bandwidth / 2
        else:
            start = freq_size // 2 - freq_size // 4
            end = freq_size // 2 + freq_size // 4
            bw = bandwidth
        size = end - start
        fc -= bw / 2
        if i % 2 == 0:
            freq1 = scipy.fft.fftshift(scipy.fft.fft(pol1[i]))[start:end]
            freq2 = scipy.fft.fftshift(scipy.fft.fft(pol2[i]))[start:end]
        else:
            freq1 = scipy.fft.fft(pol1[i])[start:end]
            freq2 = scipy.fft.fft(pol2[i])[start:end]

        # print(size,bw,fc)
        # myfreq = np.concatenate((myfreq,freq1),axis=0)
        apply_chirp(freq1, size, bw, fc)
        apply_chirp(freq2, size, bw, fc)

        p1 = scipy.fft.ifft(freq1)
        p2 = scipy.fft.ifft(freq2)

        p = np.sqrt(np.abs(p1) ** 2 + np.abs(p2) ** 2)
        fstart += psize
        psize = get_period_size(bw)
        fend = fstart + psize
        location[i] = fold_data(p, size, psize, pdata[fstart:fend], pnum[fstart:fend], location[i])
    print("coherent_dedispersion_opfb use time: ", time.time() - start_time)


def coherent_dedispersion_cspfb(pol1,pol2, nchannels, hits, profile, first_phase):
    start_time = time.time()
    freq_size = pol1.shape[1]
    num = nchannels //2
    bw = 3328 / num
    fc = 704
    # fstart = 0
    # fend = 0
    # psize = 0
    nbin = 1024
    for i in range(num):
        fc += bw / 2
        freq1 = scipy.fft.fft(pol1[i])
        freq2 = scipy.fft.fft(pol2[i])

        # print(fc)

        # freq1_backup = freq1
        # freq_size_backup = freq_size
        # bw_backup=bw
        # fc_backup = fc
        # hits_backup = hits
        # profile_backup = profile

        # apply_chirp(freq1, freq_size, bw, fc)
        apply_chirp_optimize(freq1, freq_size, bw, fc)
        apply_chirp_optimize(freq2, freq_size, bw, fc)
        # for i in range(len(freq1)):
        #     print("freq1[",i,"]",freq1[i])
        #     print("freq1_backup[",i,"]",freq1_backup[i])
        #     print("difference:",freq1[i]-freq1_backup[i])

        # apply_chirp(freq2, freq_size, bw, fc)

        fc += bw / 2
        p1 = scipy.fft.ifft(freq1)
        p2 = scipy.fft.ifft(freq2)

        p = np.sqrt(np.abs(p1) ** 2 + np.abs(p2) ** 2)
        # p = np.sqrt(np.abs(p1) ** 2)
        # fstart += psize
        # psize = get_period_size(bw)
        # fend = fstart + psize
        # location[i] = fold_data(p,freq_size,psize,pdata[fstart:fend],pnum[fstart:fend],location[i])


        # first_phase[i] = fold_integral(p, bw, first_phase[i], hits[i * nbin:(i + 1) * nbin],
        #                                profile[i * nbin:(i + 1) * nbin])
        first_phase[i] = fold_integral_optimize(p, bw, first_phase[i], hits[i * nbin:(i + 1) * nbin],
                                       profile[i * nbin:(i + 1) * nbin])
        # for i in range(len(first_phase)):
        #     print("first_phase[",i,"]",first_phase[i])
        #     print("first_phase_optimize[",i,"]",first_phase_optimize[i])
        #     print("difference:",first_phase[i]-first_phase_optimize[i])

        cur_time = time.time()
        with open('dx_pfb_cur_first_phase.txt','a') as f:
            f.write(str(i) + ":\n")
            np.savetxt(f, first_phase)

        with open('dx_pfb_cur_hits.txt','a') as f:
            f.write(str(i) + ":\n")
            np.savetxt(f, hits)

        with open('dx_pfb_cur_profile.txt','a') as f:
            f.write(str(i) + ":\n")
            np.savetxt(f, profile)
        print("save txt use time: ", time.time() - cur_time)


    print("coherent_dedispersion_cspfb use time: ", time.time() - start_time)

def coherent_dedispersion_cspfb2(pol1, pol2, nchannels, pdata, pnum, location):
    # print("pol1_result:", pol1.shape)
    # print("pol2_result:", pol2.shape)
    # print("pol1:",pol1)
    # print("pol2:",pol2)
    # print("nchannels:",nchannels)
    # print("pdata:",pdata)
    # print("pnum:",pnum)
    # print("location:",location)
    freq_size = pol1.shape[1]
    num = nchannels // 2
    bw = 400 / num
    fc = 1582
    fstart = 0
    fend = 0
    psize = 0
    for i in range(num):
        fc -= bw / 2
        freq1 = scipy.fft.fft(pol1[i])
        freq2 = scipy.fft.fft(pol2[i])

        # print("bw,fc:\n",bw,fc)
        # print("freq1 before chirp:\n", freq1)
        # print("freq2 before chirp:\n", freq2)
        apply_chirp_optimize(freq1, freq_size, bw, fc)
        apply_chirp_optimize(freq2, freq_size, bw, fc)
        # print("freq1 after chirp:\n", freq1)
        # print("freq2 after chirp:\n", freq2)

        fc -= bw / 2
        p1 = scipy.fft.ifft(freq1)
        p2 = scipy.fft.ifft(freq2)

        # print("p1:", p1)
        # print("p2:", p2)

        p = np.sqrt(np.abs(p1) ** 2 + np.abs(p2) ** 2)
        fstart += psize
        psize = get_period_size(bw)
        fend = fstart + psize
        # print("p:", p)
        # print("fstart:", fstart)
        # print("fend:", fend)
        location[i] = fold_data(p, freq_size, psize, pdata[fstart:fend], pnum[fstart:fend], location[i])
