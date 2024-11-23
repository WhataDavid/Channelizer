# Channelizer
### For ultra bandwidth signal process ###
### Supporting: ###
- Critically sampled channelizer
- Integer-oversampled channelizer
- Rationally-oversampled channelizer
### Using them by modify M and D ###
### More detail and information please access curent project's pypi website ###
### Example: ###
```python
    np_data = np.loadtxt(r'PFB-main\PFB\mini_data.txt')
    coe = realignment_coe(TAPS, CHANNEL_NUM, D)
    filter_res = polyphase_filter_bank_with_denominator_z(np_data, coe, CHANNEL_NUM, D)
    rotate_res = circular_rotate(filter_res, CHANNEL_NUM, D)
    cut_res = cut_extra_channel_data_by_tail(np.fft.fft(np.fft.ifft(rotate_res, axis=0)), CHANNEL_NUM,D) * D / M
```

