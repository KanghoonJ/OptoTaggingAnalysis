import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.widgets as sw

import os
import numpy as np
import pandas as pd
from scipy import signal
from scipy import stats
from statsmodels.stats.multitest import multipletests
import json

from aind_ephys_utils import align, sort
from open_ephys.analysis import Session

#import analysis_funcs as af

from pathlib import Path

OLD = False
set_max_power = 0.1
ignore_after_time = 3000 # ignore any events greater than 3000 seconds
# get the attached data
data_dir = "/root/capsule/data/"
curation_jsons = []
for root, dirs, files in os.walk(data_dir):
    for dir in dirs:
        if "sorted" in dir and "spike" not in dir:
            sorting_folder = os.path.join(root, dir)
    for file in files:
        if file.endswith("opto.csv"):
            events_file = file
        if file.startswith("curation") and file.endswith(".json"):
            curation_jsons.append(os.path.join(root, file))

print(sorting_folder)

session = curation_jsons[0][len(data_dir)+8:len(data_dir)+34]

recording_asset = Path(f"/data/ecephys_{session}")
recording_clipped_folder = recording_asset / "ecephys_clipped"

recording_data = Session(recording_clipped_folder).recordnodes[0].recordings[0]

# realignment in case of timestamp issue
recording_data.add_sync_line(1,            # TTL line number
                        100,          # processor ID
                        'ProbeA-AP',  # stream name
                        main=True,    # set as the main stream
                        ignore_intervals = [(ignore_after_time * 30000, np.inf)])    

recording_data.add_sync_line(1,            # TTL line number                                       
                        100,          # processor ID
                        'ProbeA-LFP', # stream name
                        ignore_intervals = [(ignore_after_time * 2500, np.inf)])

recording_data.add_sync_line(1,            # TTL line number
                        103,          # processor ID
                        'PXIe-6341',   # stream name
                        ignore_intervals = [(ignore_after_time * 30000, np.inf)])

recording_data.compute_global_timestamps(overwrite=True)

events = recording_data.events
num_streams = len(recording_data.continuous)

laser_pulses = events[(events.processor_id == 103) & (events.line == 2) & (events.state == 1)]
laser_onset_times = np.sort(laser_pulses['timestamp'].values)
event_ids = pd.read_csv(recording_clipped_folder / f'{events_file}', index_col=0)

curation_jsons = [curation_jsons[0]];
for recording, curation_json in enumerate(curation_jsons):

    # loading the manual curation data
    with open(curation_json, 'r', ) as f:
        curation_params = json.load(f)
    noise_units = list(curation_params['labelsByUnit'].keys())
    noise_units = [eval(unit) for unit in noise_units]

    if OLD:
        probe = 'ProbeA-AP'
    else:
        probe = curation_json[127:136]
        print('Probe:' + probe)
    sorting_asset = Path(f"{sorting_folder}/sorting_precurated/experiment1_Record Node 104#Neuropix-PXI-100.{probe}_recording1")
    waveform_folder = Path(f"{sorting_folder}/postprocessed/experiment1_Record Node 104#Neuropix-PXI-100.{probe}_recording1")
    
    # get timestamps for appropriate probe
    for ind in range(num_streams):
        if probe==recording_data.continuous[ind].metadata['stream_name']:
            timestamps = recording_data.continuous[ind].timestamps

    sorting_output = si.load_extractor(sorting_asset)
    we = si.WaveformExtractor.load_from_folder(waveform_folder, with_recording=False, sorting=sorting_output)
    extremum_channels = si.get_template_extremum_channel(we)

    #quality_metrics = pd.read_csv(f"/data/ecephys_{session}_sorted-ks2_5/postprocessed/experiment1_Record Node 104#Neuropix-PXI-100.ProbeA-AP_recording1/quality_metrics/metrics.csv")

    units_to_keep = sorting_output.unit_ids.flatten()
    print('Unit number all')
    print(units_to_keep)

    #units_to_keep = sorting_output.unit_ids[sorting_output.get_property("pass_qc")].flatten()
    #units_to_keep = quality_metrics.query('isi_violations_ratio < 0.5 and amplitude_cutoff < 0.1 and presence_ratio > 0.95')
    sorting_good = sorting_output.select_units(units_to_keep)

    # prepare dataframe to store laser response data
    laser_response_metrics = pd.DataFrame({'unit_id':units_to_keep})
    noise_labels = np.where(laser_response_metrics['unit_id'].isin(noise_units), 'noise', 'good')
    laser_response_metrics['noise_label'] = noise_labels

    param_group = 'train10Hz'
    # param_group = 'train5Hz'
    print('Param_group:' + param_group)

    trial_types = np.unique(event_ids.type)

    # pre-adding all the columns that will contain lists so they can store objects
    for ind, trial_type in enumerate(trial_types):
        laser_response_metrics.insert(2+5*ind,f'{trial_type}_{param_group}_all_pVals_unpaired','')
        laser_response_metrics.insert(3+5*ind,f'{trial_type}_{param_group}_all_pVals_paired','')
        laser_response_metrics.insert(4+5*ind,f'{trial_type}_{param_group}_all_latencies','')
        laser_response_metrics.insert(5+5*ind,f'{trial_type}_{param_group}_all_jitters','')
        laser_response_metrics.insert(6+5*ind,f'{trial_type}_{param_group}_all_reliability','')

    laser_response_metrics = laser_response_metrics.astype(object)

    for ind_unit, unit in enumerate(sorting_good.unit_ids):
        sample_numbers = sorting_output.get_unit_spike_train(unit)
        unit_spike_times = timestamps[sample_numbers]
        print('current unit' + unit)
        # FIXME: dealing with sorting output sometimes being longer than open ephys timestamps
        # try:
        #     unit_spike_times = timestamps[sample_numbers]
        # except(IndexError):
        #     print('SpikeInterface samples longer than Open Ephys timestamps! Removing final samples.')
        #     sample_numbers = sample_numbers[sample_numbers<len(timestamps)]
        #     unit_spike_times = timestamps[sample_numbers]

        # save best channel
        peak_channel = int(extremum_channels[unit][2:])
        laser_response_metrics.at[ind_unit, 'peak_channel'] = peak_channel
        
        for trial_type in trial_types:
            # use trials with maximum laser power for finding tagged cells
            max_power = max(event_ids.query('type == @trial_type').power)
            max_power = set_max_power;
            this_sites = list(np.unique(event_ids.query('type == @trial_type').site))
            # print(this_sites)
            this_probe = probe[:-3]
            # print(this_probe)
            if OLD:
                tag_trials = event_ids.query('param_group == @param_group and site == @this_sites and power == @max_power and type == @trial_type')
            else:
                tag_trials = event_ids.query('param_group == @param_group and site == @this_sites and power == @max_power and type == @trial_type and emission_location == @this_probe') 
            #    tag_trials = event_ids.query('param_group == @param_group') 

            all_tag_trials_timestamps = laser_onset_times[tag_trials.index.tolist()]
            
            print(tag_trials

            duration = np.unique(tag_trials.duration)[0]
            num_pulses = np.unique(tag_trials.num_pulses)[0]
            pulse_interval = np.unique(tag_trials.pulse_interval)[0]
            total_duration = (duration*num_pulses)+(pulse_interval*num_pulses)

            # --- find best stim site by finding maximum positive response during laser train ---
            laser_total_time_range = [0, (total_duration+pulse_interval)/1000]
            min_interval = np.min(tag_trials['interval'])
            baseline_time_range = [-min_interval, 0]
            unneeded_bins, baseline_spike_counts, unneeded_ids = align.to_events(unit_spike_times, all_tag_trials_timestamps, baseline_time_range, bin_size=np.diff(baseline_time_range)[0])
            baseline_spike_rate = np.mean(baseline_spike_counts)/(baseline_time_range[-1]-baseline_time_range[0])

            # for latency calculations later, using longer baseline to get better estimate of mean + stdev
            baseline_rate_stdev = np.std(baseline_spike_counts/(baseline_time_range[-1]-baseline_time_range[0]))
            threshold_spike_rate = baseline_spike_rate + 2*baseline_rate_stdev
            # time_range_raster = [-(duration * 2) / 1000, (duration + pulse_interval) / 1000]

            # smoothing params for psth for estimating max response
            win = np.concatenate((np.zeros(3), np.ones(3)))  # square (causal)
            win = win/np.sum(win)
            bin_size = 1 # in ms
            bin_edges_full = np.arange(laser_total_time_range[0],laser_total_time_range[-1], bin_size/1000)
            
            all_responses = np.zeros(len(this_sites))
            # all_responses_o = np.zeros(len(this_sites))

            for ind_site, site in enumerate(this_sites):
                this_tag_trials = tag_trials.query('site == @site')
                this_tag_laser_event_timestamps = laser_onset_times[this_tag_trials.index.tolist()]
                # creat''e smoothed PSTH of response to entire train
                unneeded_bins, laser_spike_counts, unneeded_ids = align.to_events(unit_spike_times, this_tag_laser_event_timestamps, laser_total_time_range, bin_size=bin_size/1000)

                average_response = np.mean(laser_spike_counts,axis=1)/(bin_size/1000)
                smooth_PSTH = np.convolve(average_response, win, mode='same')
                normalised_PSTH = smooth_PSTH - baseline_spike_rate
                # only looking for greatest increase in firing rate to indicate tagging
                max_response = np.max(normalised_PSTH)
                all_responses[ind_site] = max_response

    
            best_arg = np.argmax(all_responses)
            best_site = this_sites[best_arg]
            laser_response_metrics.at[ind_unit, f'{trial_type}_{param_group}_best_site'] = int(best_site)

            ### calculate latency, jitter, etc. from all pulses
            pulse_time_range = [0, duration/1000]
            pulse_offset = (duration + pulse_interval)/1000
            pulse_latency_time_range = [0, pulse_offset]

            if OLD:
                best_site_tag_trials = event_ids.query('param_group == @param_group and site == @best_site and power == @max_power and type == @trial_type')
            else:
                best_site_tag_trials = event_ids.query('param_group == @param_group and site == @best_site and power == @max_power and type == @trial_type and emission_location == @this_probe')
            best_site_event_timestamps = laser_onset_times[best_site_tag_trials.index.tolist()]

            all_pvals_unpaired = np.ones(num_pulses)
            all_pvals_paired = np.ones(num_pulses)
            all_jitter = np.zeros(num_pulses)
            all_latencies = np.zeros(num_pulses)
            all_reliability = np.zeros(num_pulses)

            # baseline counts over same time range as laser stim
            short_baseline_time_range = [-duration/1000, 0]
            unneeded_bins, short_baseline_spike_counts, unneeded_ids = align.to_events(unit_spike_times, best_site_event_timestamps, short_baseline_time_range, bin_size=np.diff(short_baseline_time_range)[0])


            # for ANOVA later
            all_pulse_spike_counts = [short_baseline_spike_counts.flatten()]

            for ind_pulse in range(num_pulses):
                this_laser_time_range = [pulse_time_range[0]+ind_pulse*pulse_offset, pulse_time_range[1]+ind_pulse*pulse_offset]
                unneeded_bins, this_pulse_spike_counts, unneeded_ids = align.to_events(unit_spike_times, best_site_event_timestamps, this_laser_time_range, bin_size=np.diff(this_laser_time_range)[0])


                all_pulse_spike_counts.append(this_pulse_spike_counts.flatten())

                # paired test
                try:
                    statistic, pVal = stats.wilcoxon(this_pulse_spike_counts.flatten()/np.diff(this_laser_time_range),
                                            short_baseline_spike_counts.flatten()/np.diff(short_baseline_time_range),
                                            alternative='greater')
                except(ValueError):  # wilcoxon test doesn't like it when there's no difference between passed values
                    statistic = 0
                    pVal = 1
                all_pvals_paired[ind_pulse] = pVal
                #laser_response_metrics.at[ind_unit, f'{trial_type}_{param_group}_pulse_{ind_pulse+1}_pVal_unpaired'] = pVal

                # unpaired test
                statistic, pVal = stats.ranksums(this_pulse_spike_counts.flatten()/np.diff(this_laser_time_range),
                                            short_baseline_spike_counts.flatten()/np.diff(short_baseline_time_range),
                                            alternative='greater')
                all_pvals_unpaired[ind_pulse] = pVal

                # reliability
                all_reliability[ind_pulse] = np.count_nonzero(this_pulse_spike_counts)/len(this_pulse_spike_counts)

                # latency
                this_latency_time_range = [pulse_latency_time_range[0]+ind_pulse*pulse_offset, pulse_latency_time_range[1]+ind_pulse*pulse_offset]
                this_bin_edges = np.arange(this_latency_time_range[0],this_latency_time_range[-1], bin_size/1000)
                
                # this_pulse_latency_locked_timestamps = af.event_locked_timestamps(unit_spike_times, best_site_event_timestamps, this_latency_time_range)
                this_pulse_latency_locked_timestamps, latency_event_ids, unneeded_ids = align.to_events(unit_spike_times, best_site_event_timestamps, this_latency_time_range)
                # this_pulse_latency_spike_counts = af.timestamps_to_spike_counts(this_pulse_latency_locked_timestamps, this_bin_edges)
                unneeded_bins, this_pulse_latency_spike_counts, unneeded_ids = align.to_events(unit_spike_times, best_site_event_timestamps, this_latency_time_range, bin_size=(bin_size/1000))
                average_response = np.mean(this_pulse_latency_spike_counts,axis=1)/(bin_size/1000)
                smooth_PSTH = np.convolve(average_response,win, mode='same')
                responsive_inds = np.flatnonzero(smooth_PSTH>threshold_spike_rate)
                if len(responsive_inds)>0:
                    first_responsive_ind = responsive_inds[0]
                    y_diff = (smooth_PSTH[first_responsive_ind]-smooth_PSTH[first_responsive_ind-1])
                    y_fraction = (threshold_spike_rate-smooth_PSTH[first_responsive_ind-1])/ y_diff
                    response_latency = this_bin_edges[first_responsive_ind] + y_fraction*(bin_size/1000) - this_latency_time_range[0]
                else:
                    response_latency = None
                all_latencies[ind_pulse] = response_latency

                # jitter
                if response_latency is not None and len(this_pulse_latency_locked_timestamps) > 0:
                    pulse_timestamps_filtered = []
                    for ind in np.unique(latency_event_ids):
                        this_pulse_first_spike_ind = np.where(latency_event_ids == ind)[0][0]
                        pulse_timestamps_filtered.append(this_pulse_latency_locked_timestamps[this_pulse_first_spike_ind])
                    # pulse_timestamps_filtered = [timestamps[0] for timestamps in this_pulse_latency_locked_timestamps if len(timestamps)>0]
                    if len(pulse_timestamps_filtered) > 1:
                        jitter = np.std(pulse_timestamps_filtered)
                    else:
                        jitter = None
                else:
                    jitter = None
                all_jitter[ind_pulse] = jitter

            # corrected pvals
            (responsive_sites_unpaired, corrected_pVals_unpaired, alphaSidak, alphaBonf) = multipletests(all_pvals_unpaired, method='holm')
            (responsive_sites_paired, corrected_pVals_paired, alphaSidak, alphaBonf) = multipletests(all_pvals_paired, method='holm')

            # calculate p-value from kruskal-wallis test (unpaired) and friedman test (paired) over all laser pulses
            try:
                statistic, pVal_unpaired = stats.kruskal(*all_pulse_spike_counts)
            except(ValueError):  # kruskal test ALSO doesn't like it when there's no difference between passed values
                statistic = 0
                pVal_unpaired = 1
            statistic, pVal_paired = stats.friedmanchisquare(*all_pulse_spike_counts)

            # save all the things
            laser_response_metrics.at[ind_unit, f'{trial_type}_{param_group}_overall_pVal_unpaired'] = pVal_unpaired
            laser_response_metrics.at[ind_unit, f'{trial_type}_{param_group}_overall_pVal_paired'] = pVal_paired

            laser_response_metrics.at[ind_unit, f'{trial_type}_{param_group}_all_pVals_unpaired'] = corrected_pVals_unpaired.tolist()
            laser_response_metrics.at[ind_unit, f'{trial_type}_{param_group}_all_pVals_paired'] = corrected_pVals_paired.tolist()
            laser_response_metrics.at[ind_unit, f'{trial_type}_{param_group}_all_latencies'] = all_latencies.tolist()
            laser_response_metrics.at[ind_unit, f'{trial_type}_{param_group}_all_jitters'] = all_jitter.tolist()
            laser_response_metrics.at[ind_unit, f'{trial_type}_{param_group}_all_reliability'] = all_reliability.tolist()

            laser_response_metrics.at[ind_unit, f'{trial_type}_{param_group}_mean_latency'] = np.mean(all_latencies)
            laser_response_metrics.at[ind_unit, f'{trial_type}_{param_group}_mean_jitter'] = np.mean(all_jitter)
            laser_response_metrics.at[ind_unit, f'{trial_type}_{param_group}_mean_reliability'] = np.mean(all_reliability)

            laser_response_metrics.at[ind_unit, f'{trial_type}_{param_group}_num_sig_pulses_unpaired'] = np.sum(responsive_sites_unpaired)
            laser_response_metrics.at[ind_unit, f'{trial_type}_{param_group}_num_sig_pulses_paired'] = np.sum(responsive_sites_paired)

            #### compare peak channel to peak emission site
            peak_channel_diff = abs(peak_channel - 10*best_site)
            laser_response_metrics.at[ind_unit, f'{trial_type}_{param_group}_channel_diff'] = int(peak_channel_diff)

    laser_response_metrics.to_csv(f'/results/{session}_{probe}_laser_response_metrics.csv')
