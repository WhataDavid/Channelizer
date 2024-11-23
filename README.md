# Channelizer
### For ultra bandwidth signal process ###
### Supporting Critically sampled channelizer / Integer-oversampled channelizer / Rationally-oversampled channelizer ###
### Using them by modify M and D ###
### More detail and information please access curent project's pypi website ###
···
    dx_ospfb_coe = realignment_coe(TAPS, CHANNEL_NUM, D)
    dx_ospfb_out = polyphase_filter_bank_with_denominator_z(np_data, dx_ospfb_coe, CHANNEL_NUM, D)
    dx_ospfb_rotate = circular_rotate(dx_ospfb_out, CHANNEL_NUM, D)
    dx_ospfb_cut = cut_extra_channel_data_by_tail(np.fft.fft(np.fft.ifft(dx_ospfb_rotate, axis=0)), CHANNEL_NUM,
                                                  D) * D / M
···
