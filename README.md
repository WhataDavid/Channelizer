# Channelizer
### For ultra bandwidth signal process ###
### Supporting: ###
- Critically sampled channelizer
- Integer-oversampled channelizer
- Rationally-oversampled channelizer
### Using them by modify M and D ###
### More detail and information please access curent project's pypi website ###
### Example: ###
```
    coe = realignment_coe(TAPS, CHANNEL_NUM, D)
    filter_res = polyphase_filter_bank_with_denominator_z(np_data, dx_ospfb_coe, CHANNEL_NUM, D)
    rotate_res = circular_rotate(dx_ospfb_out, CHANNEL_NUM, D)
    cut_res = cut_extra_channel_data_by_tail(np.fft.fft(np.fft.ifft(dx_ospfb_rotate, axis=0)), CHANNEL_NUM,
                                               D) * D / M
```

