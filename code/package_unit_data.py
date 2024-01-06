import numpy as np

def package_unit_data(good_cells, sorting_output, timestamps, laser_onset_times, event_ids, waveform_extractor, session, probe, red_tag, blue_tag, damn_chrmine=False):
    # data we'll be collecting
    all_amplitudes = np.empty(len(good_cells))
    all_waveforms = None
    all_ISIs = None
    all_baselines = np.empty(len(good_cells))

    param_group = 'train5Hz'
    train_trials = event_ids.query('param_group == @param_group')
    duration = np.unique(train_trials.duration)[0]
    num_pulses = np.unique(train_trials.num_pulses)[0]
    pulse_interval = np.unique(train_trials.pulse_interval)[0]
    total_duration = ((duration*num_pulses)+(pulse_interval*(num_pulses-1))) / 1000
    time_range = [-max(total_duration * 0.5, 0.1), total_duration + max(total_duration * 0.5, 0.1)]

    trial_types = np.unique(event_ids.type)

    for ind_unit, unit in enumerate(good_cells.unit_id):
        sample_numbers = sorting_output.get_unit_spike_train(unit)
        unit_spike_times = timestamps[sample_numbers]
        unit_waveform = waveform_extractor.get_template(unit)
        unit_metrics = good_cells.query('unit_id == @unit')

        # save the aggregate data we'll be plotting later!
        # normalised waveform
        peak_channel = int(unit_metrics['peak_channel'])
        peak_waveform = unit_waveform[70:170,peak_channel]
        unit_amplitude = -np.min(peak_waveform)
        normed_waveform = peak_waveform/unit_amplitude
        if all_waveforms is None:
            all_waveforms = np.empty((len(good_cells), len(normed_waveform)))
        all_waveforms[ind_unit, :] = normed_waveform
        all_amplitudes[ind_unit] = unit_amplitude

        # get ISI histograms
        bin_size = 0.002
        ISI_bins = np.arange(0, 0.2 + bin_size, bin_size)
        # only want ISIs when there was no laser presentation!
        first_event = laser_onset_times[0]
        if first_event < unit_spike_times[0]:
            all_ISI = np.nan
        else:
            spike_ind_first_event = np.searchsorted(unit_spike_times, first_event)
            spike_times_before_event = unit_spike_times[0:spike_ind_first_event]
            all_ISI = np.diff(spike_times_before_event)

        if all_ISIs is None:
            all_ISIs = np.empty((len(good_cells), len(ISI_bins)-1))
        all_ISIs[ind_unit, :] = np.histogram(all_ISI, ISI_bins, density=True)[0]

        # get baseline firing rate
        if first_event < unit_spike_times[0]:
            all_baselines[ind_unit] = np.nan
        else:
            num_spikes = len(spike_times_before_event)
            pre_stim_duration = spike_times_before_event[-1]-spike_times_before_event[0]
            all_baselines[ind_unit] = num_spikes/pre_stim_duration

    tagged_unit_blue = (good_cells['internal_blue_train_num_sig_pulses_unpaired'] == 5) & (good_cells['internal_blue_train_channel_diff'] < 20) & (good_cells['internal_blue_train_mean_latency'] < 0.007)
    tagged_unit_red = (good_cells['internal_red_train_num_sig_pulses_unpaired'] == 5) & (good_cells['internal_red_train_channel_diff'] < 20) & (good_cells['internal_red_train_mean_latency'] < 0.007)
    tagged_both = tagged_unit_blue & tagged_unit_red

    unit_tags = np.tile('unknown', len(good_cells))
    unit_tags[tagged_unit_blue] = blue_tag
    unit_tags[tagged_unit_red] = red_tag

    if not damn_chrmine:
        unit_tags[tagged_both] = f'{blue_tag} and {red_tag}'

    # tagged_unit_blue_surface = (good_cells['external_blue_train_num_sig_pulses_unpaired'] == 5) & (good_cells['external_blue_train_mean_latency'] < 0.007)
    # tagged_unit_red_surface = None

    # save aggregated stats
    outputFname = f'/results/{session}_{probe}_tagged_units.npz'
    np.savez(outputFname,
            amplitudes=all_amplitudes,
            depths=good_cells.peak_channel,
            waveforms=all_waveforms, 
            ISIhist=all_ISIs, 
            ISI_bins=ISI_bins, 
            firing_rates=all_baselines,
            unit_spike_timestamps=unit_spike_times[unit_spike_times<first_event],
            unit_tags=unit_tags)
    print(outputFname + " saved")

    return all_amplitudes, all_baselines, tagged_unit_blue, tagged_unit_red