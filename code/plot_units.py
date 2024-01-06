import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.widgets as sw

import os
import numpy as np
import pandas as pd
import ast

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import analysis_funcs as af
import plotting_funcs as pf

def plot_opto_stat_summaries(all_amplitudes, all_firing_rates, tagged_unit_blue, tagged_unit_red, good_cells, session, probe, fig_format='pdf', annotate_points=False):
    # plot a bunch of stuff about all the units
    plt.clf()
    gs_summary = gridspec.GridSpec(2, 2)
    gs_summary.update(top=0.93, bottom=0.05, left=0.08, right=0.95, wspace=0.2, hspace=0.4)

    # plot summary of depth vs amplitude with labeling for this session
    xvals = all_amplitudes
    yvals = np.array(good_cells.peak_channel)
    ax_amp = plt.subplot(gs_summary[0, 0])
    plt.scatter(xvals, yvals, facecolors='0.75', edgecolors='none')
    # colour the tagged units, put them on top
    plt.scatter(xvals[tagged_unit_red], yvals[tagged_unit_red], facecolors='tomato', edgecolors='none', zorder=10)
    plt.scatter(xvals[tagged_unit_blue], yvals[tagged_unit_blue], facecolors='deepskyblue', edgecolors='none', zorder=10)
    #plt.scatter(xvals[tagged_both], yvals[tagged_both], facecolors='violet', edgecolors='none', zorder=20)
    if annotate_points:
        point_labels = good_cells.unit_id.tolist()
        for ind, txt in enumerate(point_labels):
            ax_amp.annotate(txt, (xvals[ind], yvals[ind]))
    plt.xlim(0,550)
    plt.ylim(0,150)
    plt.xlabel('Amplitude (mV)')
    plt.ylabel('Depth (Channel number)')
    legend_elements = [Line2D([0], [0], marker='o', markerfacecolor='0.75', label='Untagged', markeredgecolor='w'),
                    # Line2D([0], [0], marker='o', markeredgecolor='b', label='Surface Responsive', markerfacecolor='0.75'),
                    Line2D([0], [0], marker='o', markerfacecolor='deepskyblue', label='NPopto Responsive', markeredgecolor='w')]
    plt.legend(handles=legend_elements, loc='best', handlelength=0)
    ax_amp.spines['top'].set_visible(False)
    ax_amp.spines['right'].set_visible(False)

    # plot summary of latency for tagged and untagged units
    xvals_untagged_blue = 1000*np.array(good_cells[~tagged_unit_blue].internal_blue_train_mean_latency.dropna())
    xvals_tagged_blue = 1000*np.array(good_cells[tagged_unit_blue].internal_blue_train_mean_latency)
    #xvals_tagged_red = np.array(good_cells[tagged_unit_red])
    ax_latency = plt.subplot(gs_summary[0, 1])

    bins = np.arange(30)
    plt.hist([xvals_untagged_blue, xvals_tagged_blue], bins, label=['untagged', 'tagged'], color=['0.75', 'deepskyblue'])
    plt.legend(loc='upper right')
    plt.xlabel('Laser response latency (ms)')
    plt.ylabel('Count')
    ax_latency.spines['top'].set_visible(False)
    ax_latency.spines['right'].set_visible(False)

    # plot summary of reliability for tagged and untagged units
    xvals_untagged_blue = 100*np.array(good_cells[~tagged_unit_blue].internal_blue_train_mean_reliability.dropna())
    xvals_tagged_blue = 100*np.array(good_cells[tagged_unit_blue].internal_blue_train_mean_reliability)
    #xvals_tagged_red = np.array(good_cells[tagged_unit_red])
    ax_reliability = plt.subplot(gs_summary[1, 0])

    bins = np.arange(0,105,5)
    plt.hist([xvals_untagged_blue, xvals_tagged_blue], bins, label=['untagged', 'tagged'], color=['0.75', 'deepskyblue'])
    plt.legend(loc='upper right')
    plt.xlabel('Percentage of trials with spikes during laser presentation')
    plt.ylabel('Count')
    ax_reliability.spines['top'].set_visible(False)
    ax_reliability.spines['right'].set_visible(False)

    # plot summary of jitter for tagged and untagged units
    xvals_untagged_blue = 1000*np.array(good_cells[~tagged_unit_blue].internal_blue_train_mean_jitter.dropna())
    xvals_untagged_blue = xvals_untagged_blue[xvals_untagged_blue!=0]
    xvals_tagged_blue = 1000*np.array(good_cells[tagged_unit_blue].internal_blue_train_mean_jitter)
    #xvals_tagged_red = np.array(good_cells[tagged_unit_red])
    ax_jitter = plt.subplot(gs_summary[1, 1])

    bins = np.arange(20)
    plt.hist([xvals_untagged_blue, xvals_tagged_blue], bins, label=['untagged', 'tagged'], color=['0.75', 'deepskyblue'])
    plt.legend(loc='upper right')
    plt.xlabel('STDev of first spike time (ms)')
    plt.ylabel('Count')
    ax_jitter.spines['top'].set_visible(False)
    ax_jitter.spines['right'].set_visible(False)


    # save figure
    plt.gcf().set_size_inches((10,10))
    plt.suptitle(f'{session} {probe}')
    fig_name = f'{session}_{probe}_tagging_summary.{fig_format}'    
    print(os.getcwd())
    plt.savefig(f'../results/{fig_name}', format=fig_format)
    print(f'{fig_name} saved')

def plot_all_good_units(units_to_plot, sorting_output, timestamps, laser_onset_times, event_ids, waveform_extractor, session, probe, max_channel, param_group, fig_format='png', fig_name=None, old_tag=False):
    #num_rows = int(np.ceil(np.sqrt(len(units_to_plot))))
    #num_cols = int(np.ceil(len(units_to_plot) / num_rows))

    num_cols = int(np.ceil(np.sqrt(len(units_to_plot))))
    num_rows = int(np.ceil(len(units_to_plot) / num_cols))

    trial_types = np.unique(event_ids.type)

    plt.clf()
    gs = gridspec.GridSpec(num_rows, num_cols)
    gs.update(top=0.93, bottom=0.05, left=0.03, right=0.97, wspace=0.2, hspace=0.5)

    # format the gridspec based on trial types present
    if ('external_blue' or 'external_red') in trial_types:
        height_ratios=[0.01,0.3,1]
        hspace = 0.8
        panel_height = 5
    else:
        height_ratios=[0.005,1]
        hspace = 0.5
        panel_height = 3

    #width_ratios=np.ones(len(np.unique(event_ids['wavelength']))+1).tolist()
    width_ratios = [1,1,1] # fix this later

    print(f'Plotting {len(units_to_plot.unit_id)} units')

    for ind_unit, unit in enumerate(units_to_plot.unit_id):
        sample_numbers = sorting_output.get_unit_spike_train(unit)
        unit_spike_times = timestamps[sample_numbers]
        unit_waveform = waveform_extractor.get_template(unit)
        print(f'Unit {ind_unit}')
        print(trial_types)        
        gs_this_unit = gridspec.GridSpecFromSubplotSpec(1 + 2//2, 3, subplot_spec=gs[ind_unit//num_cols, np.mod(ind_unit, num_cols)], wspace=0.3, hspace=hspace, height_ratios=height_ratios, width_ratios=width_ratios)

        for ind_type, trial_type in enumerate(trial_types):
            if 'internal' in trial_type:
                # plot npopto stim by site
                max_power = max(event_ids.query('type == @trial_type').power)
                sites = list(np.unique(event_ids.query('type == @trial_type').site))
                if old_tag:
                    tag_trials = event_ids.query('param_group == @param_group and site == @sites and power == @max_power and type == @trial_type')
                else:
                    tag_trials = event_ids.query('param_group == @param_group and site == @sites and power == @max_power and type == @trial_type and emission_location == @probe')
                y_axis = tag_trials.site.tolist()
                y_label = 'Emission site'
                y_ticks = sites
                x_label = 'Time from laser onset (s)'

            elif 'external' in trial_type:
                # plot surface stim by power
                site = list(np.unique(event_ids.query('type == @trial_type').site))
                powers = list(np.unique(event_ids.query('type == @trial_type').power))
                tag_trials = event_ids.query('param_group == @param_group and site == @site and power == @powers and type == @trial_type')
                y_axis = tag_trials.power.tolist()
                y_label = 'Laser power (mW)'
                y_ticks = powers
                x_label = None

            duration = np.unique(tag_trials.duration)[0]
            num_pulses = np.unique(tag_trials.num_pulses)[0]
            pulse_interval = np.unique(tag_trials.pulse_interval)[0]
            total_duration = (duration*num_pulses)+(pulse_interval*num_pulses)
            raster_time_range = [-total_duration/1000, (2*total_duration)/1000]
            wavelength = np.unique(tag_trials.wavelength)[0]

            ax_raster = plt.subplot(gs_this_unit[1 + ind_type//2, ind_type%2])
            this_event_timestamps = laser_onset_times[tag_trials.index.tolist()]
            event_locked_timestamps = af.event_locked_timestamps(unit_spike_times, this_event_timestamps, raster_time_range)
            pf.raster_plot(event_locked_timestamps, raster_time_range, y_axis, ms=100, markeredgecolor='none')
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.xlim(raster_time_range)
            ax_raster.set_yticklabels(y_ticks)

            this_unit_id = units_to_plot.unit_id.tolist()[ind_unit]
            # this_type_laser_param = f'{trial_type}_{param_group}_num_sig_pulses_paired'
            # weight = 'extra bold' if units_to_plot[this_type_laser_param].iloc[ind_unit] == 5 else 'normal'
            # title_color = 'red' if units_to_plot[this_type_laser_param].iloc[ind_unit] == 5 else 'black'
            # plt.title(f'{trial_type}', fontweight = weight, color = title_color)
            plt.title(f'{trial_type}, {wavelength} nm')


            # patches showing laser presentation
            yLims = np.array(plt.ylim())
            laser_color = 'tomato' if 'red' in trial_type else 'skyblue'
            for pulse in range(num_pulses):
                rect = patches.Rectangle((pulse * (duration+pulse_interval)/1000, yLims[0]), duration / 1000, yLims[1] - yLims[0], linewidth=1, edgecolor=laser_color, facecolor=laser_color, alpha=0.35, clip_on=False)
                ax_raster.add_patch(rect)
        print('Plotted rasters')

        # plot waveform
        ax_waveform = plt.subplot(gs_this_unit[1:,2])
        cmap = matplotlib.cm.PRGn
        #cmap = matplotlib.cm.viridis
        shifted_cmap = pf.shiftedColorMap(cmap, np.min(unit_waveform[70:170,:]), np.max(unit_waveform[70:170,:]), 'shifted_PRGn')
        waveform_peak = np.min(unit_waveform)
        plt.imshow(0.195*unit_waveform[70:170,:max_channel].T, aspect='auto', cmap=shifted_cmap)
        ax_waveform.invert_yaxis()
        plt.ylabel('Channel')
        plt.xlabel('Sample number')
        cbar = plt.colorbar()
        cbar.set_label('Voltage (mV)')

        # inset with peak channel waveform
        peak_ind = np.where(unit_waveform==waveform_peak)
        peak_waveform = unit_waveform[70:150,peak_ind[1][0]]
        #peak_channel = int(units_to_plot.iloc[ind_unit]['peak_channel']) - 1
        #print('Peak Ch: ', peak_channel)
        #peak_waveform = unit_waveform[70:170,peak_channel]
                


        ax_inset = inset_axes(ax_waveform, width="30%", height="25%", loc=1, bbox_to_anchor=(0, 0, 1, 1), bbox_transform=ax_waveform.transAxes)
        plt.plot(peak_waveform, lw=2, c='k')
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        print('Plotted waveforms')

        # Add ghost axes so I can add a title with cluster number...
        ax_title = plt.subplot(gs_this_unit[0, :])
        ax_title.axis('off')
        ax_title.set_title(f'cluster {this_unit_id}', fontsize=16)
        print('Added title')

    plt.gcf().set_size_inches((11 * num_cols, panel_height * num_rows))
    plt.suptitle(f'{session}, {probe}', fontsize=20)
    if fig_name is None:
        fig_name = f'{session}_{probe}_all_good_units.{fig_format}'
    print(os.getcwd())

    plt.savefig(f'../results/{fig_name}', format=fig_format)
    print(f'{fig_name} saved')

def plot_laser_responsive_units(units_to_plot, sorting_output, timestamps, laser_onset_times, event_ids, session, probe, param_group, fig_format='png', fig_name=None, old_tag=False):
    # plot some interesting units
    #num_rows = int(np.ceil(np.sqrt(len(units_to_plot))))
    #num_cols = int(np.ceil(len(units_to_plot) / num_rows))

    num_cols = int(np.ceil(np.sqrt(len(units_to_plot))))
    num_rows = int(np.ceil(len(units_to_plot) / num_cols))

    plt.clf()
    gs_cool_units = gridspec.GridSpec(num_rows, num_cols)
    gs_cool_units.update(top=0.90, bottom=0.08, left=0.03, right=0.97, wspace=0.2, hspace=0.7)

    # format the gridspec based on trial types present
    if not old_tag:
        event_ids = event_ids.query('emission_location == @probe')
    width_ratios=np.ones(len(np.unique(event_ids['wavelength']))).tolist()

    trial_types = np.unique(event_ids.type)
    internal_trials = [trial_type for trial_type in trial_types if 'internal' in trial_type]

    for ind_unit, unit in enumerate(units_to_plot.unit_id):
        gs_this_cool_unit = gridspec.GridSpecFromSubplotSpec(2, len(width_ratios), subplot_spec=gs_cool_units[ind_unit//num_cols, np.mod(ind_unit, num_cols)], wspace=0.2, hspace=0.3, height_ratios=[0.01,1], width_ratios=width_ratios)

        sample_numbers = sorting_output.get_unit_spike_train(unit)
        unit_spike_times = timestamps[sample_numbers]
        this_unit_id = units_to_plot.unit_id.tolist()[ind_unit]

        # plot raster of laser response for each pulse and each laser colour
        for ind_type, trial_type in enumerate(internal_trials):
            max_power = max(event_ids.query('type == @trial_type').power)
            best_site = units_to_plot[f'{trial_type}_{param_group}_best_site'].iloc[ind_unit]
            tag_trials = event_ids.query('param_group == @param_group and site == @best_site and power == @max_power and type == @trial_type')
            duration = np.unique(tag_trials.duration)[0]
            num_pulses = np.unique(tag_trials.num_pulses)[0]
            pulse_interval = np.unique(tag_trials.pulse_interval)[0]
            total_duration = (duration*num_pulses)+(pulse_interval*num_pulses)
            time_range_raster = [-(duration * 2) / 1000, (duration + pulse_interval) / 1000]
            
            int_event_locked_timestamps = []
            pulse_nums = []
            this_event_timestamps = laser_onset_times[tag_trials.index.tolist()]

            for pulse_num in range(num_pulses):
                time_shift = pulse_num*(duration+pulse_interval)/1000
                this_time_range = [time_range_raster[0]+time_shift, time_range_raster[1]+time_shift]
                this_event_locked_timestamps = af.event_locked_timestamps(unit_spike_times, this_event_timestamps, this_time_range, time_shift=time_shift)
                int_event_locked_timestamps.extend(this_event_locked_timestamps)
                pulse_nums.extend([pulse_num+1]*len(this_event_locked_timestamps))

            ax_raster = plt.subplot(gs_this_cool_unit[1, ind_type])
            pf.raster_plot(int_event_locked_timestamps, time_range_raster, cond_each_trial=pulse_nums, ms=100)
            plt.xlabel('Time from laser onset (s)')
            if ind_type==0:
                plt.ylabel('Pulse number')
            plt.xlim(time_range_raster)

            # show latency for each pulse
            this_type_latency_param = f'{trial_type}_{param_group}_all_latencies'
            all_latencies = units_to_plot[this_type_latency_param].iloc[ind_unit]
            # pulling some janky shit to read a list stored in the dataframe...
            all_latencies = all_latencies.replace('nan', 'None')
            all_latencies = ast.literal_eval(all_latencies)
            figure_cond_fraction = 1/num_pulses
            for pulse_num in range(num_pulses):
                this_pulse_latency = all_latencies[pulse_num]
                if this_pulse_latency is not None:
                    plt.axvline(this_pulse_latency, ymin=pulse_num*figure_cond_fraction, ymax=(pulse_num+1)*figure_cond_fraction, color='gray')

            # patches showing laser presentation
            yLims = np.array(plt.ylim())
            laser_color = 'skyblue' if 'blue' in trial_type else 'tomato'
            rect = patches.Rectangle((0, yLims[0]), duration / 1000, yLims[1] - yLims[0], linewidth=1, edgecolor=laser_color,
                                        facecolor=laser_color, alpha=0.15, clip_on=False)
            ax_raster.add_patch(rect)

            this_type_laser_param = f'{trial_type}_{param_group}_num_sig_pulses_paired'
            this_type_latency_param = f'{trial_type}_{param_group}_mean_latency'
            #tagged_unit = units_to_plot[this_type_laser_param].iloc[ind_unit] == 5
            tagged_unit = (units_to_plot[this_type_laser_param].iloc[ind_unit] == 5 and units_to_plot[this_type_latency_param].iloc[ind_unit] < 0.007)
            weight = 'extra bold' if tagged_unit else 'normal'
            title_color = 'red' if tagged_unit else 'black'
            plt.title(f'{trial_type}, site {best_site:1.0f}', fontweight = weight, color = title_color)

        ax_title = plt.subplot(gs_this_cool_unit[0, :])
        ax_title.axis('off')
        ax_title.set_title(f'cluster {this_unit_id}')

    plt.gcf().set_size_inches((5 * num_cols, 2.5 * num_rows))
    plt.suptitle(f'{session}_{probe}')
    if fig_name is None:
        fig_name = f'{session}_{probe}_interesting_units.{fig_format}'
    plt.savefig(f'../results/{fig_name}', format=fig_format)
    print(f'{fig_name} saved')

def plot_ROI_unit(units_to_plot, sorting_output, timestamps, laser_onset_times, event_ids, waveform_extractor, session, probe, max_channel, param_group, fig_format='png', fig_name=None, old_tag=False):
    plt.clf()
    for ind_unit, unit in enumerate(units_to_plot.unit_id):
        sample_numbers = sorting_output.get_unit_spike_train(unit)
        unit_spike_times = timestamps[sample_numbers]
        unit_waveform = waveform_extractor.get_template(unit)
        print(f'Unit {ind_unit}')
        trial_types = np.unique(event_ids.type)
        print(trial_types)        
        # gs_this_unit = gridspec.GridSpecFromSubplotSpec(1 + 2//2, 3, subplot_spec=gs[ind_unit//num_cols, np.mod(ind_unit, num_cols)], wspace=0.3, hspace=hspace, height_ratios=height_ratios, width_ratios=width_ratios)

        # for ind_type, trial_type in enumerate(trial_types):
        #     if 'internal' in trial_type:
        #         # plot npopto stim by site
        #         max_power = max(event_ids.query('type == @trial_type').power)
        #         sites = list(np.unique(event_ids.query('type == @trial_type').site))
        #         if old_tag:
        #             tag_trials = event_ids.query('param_group == @param_group and site == @sites and power == @max_power and type == @trial_type')
        #         else:
        #             tag_trials = event_ids.query('param_group == @param_group and site == @sites and power == @max_power and type == @trial_type and emission_location == @probe')
        #         y_axis = tag_trials.site.tolist()
        #         y_label = 'Emission site'
        #         y_ticks = sites
        #         x_label = 'Time from laser onset (s)'

        #     elif 'external' in trial_type:
        #         # plot surface stim by power
        #         site = list(np.unique(event_ids.query('type == @trial_type').site))
        #         powers = list(np.unique(event_ids.query('type == @trial_type').power))
        #         tag_trials = event_ids.query('param_group == @param_group and site == @site and power == @powers and type == @trial_type')
        #         y_axis = tag_trials.power.tolist()
        #         y_label = 'Laser power (mW)'
        #         y_ticks = powers
        #         x_label = None

        tag_trials = event_ids.query('param_group == @param_group and site == @sites and power == @max_power and type == @trial_type and emission_location == @probe')

        duration = np.unique(tag_trials.duration)[0]
        num_pulses = np.unique(tag_trials.num_pulses)[0]
        pulse_interval = np.unique(tag_trials.pulse_interval)[0]
        total_duration = (duration*num_pulses)+(pulse_interval*num_pulses)
        raster_time_range = [-total_duration/1000, (2*total_duration)/1000]
        wavelength = np.unique(tag_trials.wavelength)[0]

        ax_raster = plt.subplot(gs_this_unit[1 + ind_type//2, ind_type%2])
        this_event_timestamps = laser_onset_times[tag_trials.index.tolist()]
        event_locked_timestamps = af.event_locked_timestamps(unit_spike_times, this_event_timestamps, raster_time_range)
        pf.raster_plot(event_locked_timestamps, raster_time_range, y_axis, ms=100, markeredgecolor='none')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xlim(raster_time_range)
        ax_raster.set_yticklabels(y_ticks)

        this_unit_id = units_to_plot.unit_id.tolist()[ind_unit]
        # this_type_laser_param = f'{trial_type}_{param_group}_num_sig_pulses_paired'
        # weight = 'extra bold' if units_to_plot[this_type_laser_param].iloc[ind_unit] == 5 else 'normal'
        # title_color = 'red' if units_to_plot[this_type_laser_param].iloc[ind_unit] == 5 else 'black'
        # plt.title(f'{trial_type}', fontweight = weight, color = title_color)
        # plt.title(f'{trial_type}, {wavelength} nm')


        # patches showing laser presentation
        yLims = np.array(plt.ylim())
        laser_color = 'tomato' # if 'red' in trial_type else 'skyblue'
        for pulse in range(num_pulses):
            rect = patches.Rectangle((pulse * (duration+pulse_interval)/1000, yLims[0]), duration / 1000, yLims[1] - yLims[0], linewidth=1, edgecolor=laser_color, facecolor=laser_color, alpha=0.35, clip_on=False)
            ax_raster.add_patch(rect)
        print('Plotted rasters')

        # plot waveform
        ax_waveform = plt.subplot(1,1)
        cmap = matplotlib.cm.PRGn
        #cmap = matplotlib.cm.viridis
        shifted_cmap = pf.shiftedColorMap(cmap, np.min(unit_waveform[70:170,:]), np.max(unit_waveform[70:170,:]), 'shifted_PRGn')
        waveform_peak = np.min(unit_waveform)
        plt.imshow(0.195*unit_waveform[70:170,:max_channel].T, aspect='auto', cmap=shifted_cmap)
        ax_waveform.invert_yaxis()
        plt.ylabel('Channel')
        plt.xlabel('Sample number')
        cbar = plt.colorbar()
        cbar.set_label('Voltage (mV)')

        # inset with peak channel waveform
        peak_ind = np.where(unit_waveform==waveform_peak)
        peak_waveform = unit_waveform[70:150,peak_ind[1][0]]
        #peak_channel = int(units_to_plot.iloc[ind_unit]['peak_channel']) - 1
        #print('Peak Ch: ', peak_channel)
        #peak_waveform = unit_waveform[70:170,peak_channel]
                


        ax_inset = inset_axes(ax_waveform, width="30%", height="25%", loc=1, bbox_to_anchor=(0, 0, 1, 1), bbox_transform=ax_waveform.transAxes)
        plt.plot(peak_waveform, lw=2, c='k')
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        print('Plotted waveforms')

        # Add ghost axes so I can add a title with cluster number...
        ax_title = plt.subplot(1,2)
        ax_title.axis('off')
        ax_title.set_title(f'cluster {this_unit_id}', fontsize=16)
        print('Added title')

    # plt.gcf().set_size_inches((11 * num_cols, panel_height * num_rows))
    plt.suptitle(f'{session}, {probe}', fontsize=20)
    if fig_name is None:
        fig_name = f'{session}_{probe}_all_good_units.{fig_format}'
    print(os.getcwd())

    # plt.savefig(f'../results/{fig_name}', format=fig_format)
    # print(f'{fig_name} saved')