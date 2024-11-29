# Channelizer
### For ultra bandwidth signal process ###
### Supporting: ###
- Critically sampled channelizer
- Integer-oversampled channelizer
- Rationally-oversampled channelizer
### Using them by modify M and D ###
### Code Example: ###
```python
    import channelizers

    # filter taps:
    TAPS = 63
    # channel_num(branch), Number of frequency bands :
    CHANNEL_NUM = 4
    # M equal channel_num(branch), more article call it M:
    M = CHANNEL_NUM
    # Decimation factor
    D = 3
    # if D = M : Critical polyphase filter bank(CSPFB)
    # if M is multiples of D : Integer-oversample filter bank(IOSPFB)
    # else if D < M : Rationally-oversampled filter bank(ROSPFB)

    np_data = np.loadtxt(r'\mini_data.txt')
    coe = realignment_coe(TAPS, CHANNEL_NUM, D)
    polyphase_filter_res = polyphase_filter_bank_with_denominator_z(np_data, coe, CHANNEL_NUM, D)
    fft_res = np.fft.ifft(polyphase_filter_res, axis=0)
    plot_sub(fft_res, CHANNEL_NUM, "DX " + str(CHANNEL_NUM) + "/" + str(D) + "X channelizer with z gcd result:")
    
    coe = realignment_coe(TAPS, CHANNEL_NUM, D)
    polyphase_filter_res = polyphase_filter_bank_with_denominator_z(np_data, coe, CHANNEL_NUM, D)
    rotate_res = circular_rotate(polyphase_filter_res, CHANNEL_NUM, D)
    fft_res = np.fft.ifft(rotate_res, axis=0)
    plot_sub(fft_res, CHANNEL_NUM,"DX " + str(CHANNEL_NUM) + "/" + str(D) + "X channelizer with z gcd and rotate result:")
    
    coe = realignment_coe(TAPS, CHANNEL_NUM, D)
    polyphase_filter_res = polyphase_filter_bank_with_denominator_z(np_data, coe, CHANNEL_NUM, D)
    rotate_res = circular_rotate(polyphase_filter_res, CHANNEL_NUM, D)
    cut_res = cut_extra_channel_data_by_tail(np.fft.fft(np.fft.ifft(rotate_res, axis=0)), CHANNEL_NUM,D) * D / M
    plot_sub(np.fft.ifft(cut_res), CHANNEL_NUM,"DX " + str(CHANNEL_NUM) + "/" + str(D) + "X channelizer with z gcd rotate and cut result:")
```
### Install ###
```pip install channelizers```

Or clone code
### More detail and information please access curent project's pypi website ###
