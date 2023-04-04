import seaborn as sns
import mne
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mne.stats import permutation_cluster_1samp_test as pcluster_test
from mne.viz.utils import center_cmap
from copy import deepcopy

def get_EpochsTF(epochs, want_chs, tmin, tmax,  l_freq=2, h_freq=36
                 , sfreq=None, tbase=0, apply_baseline=True):
    _epochs = deepcopy(epochs)
    freqs = np.arange(l_freq, h_freq, 1)
    _epochs.load_data()
    _epochs.pick_channels(want_chs)
    if sfreq is not None:
        _epochs.resample(sfreq=sfreq)

    n_cycles = freqs  # use constant t/f resolution
    baseline = [tmin, tbase]  # baseline interval (in s)
    # Run TF decomposition overall epochs
    tfr = mne.time_frequency.tfr_multitaper(_epochs, freqs=freqs, n_cycles=n_cycles,
                         use_fft=True, return_itc=False, average=False,decim=2)
    tfr.crop(tmin, tmax)
    if apply_baseline:
        tfr.apply_baseline(baseline, mode="percent")

    df = tfr.to_data_frame(time_format=None, long_format=True)
    return tfr, df

def draw_band_EpochsTF(tfr_df, want_chs, vmin=-1.5, vmax_ratio=3):
    vmax = tfr_df['value'].to_numpy().std() * vmax_ratio
    # Map to frequency bands:
    freq_bounds = {'_': 0,
                   'delta': 3,
                   'theta': 7,
                   'alpha': 13,
                   'beta': 35,
                   'gamma': 140}
    tfr_df['band'] = pd.cut(tfr_df['freq'], list(freq_bounds.values()),
                        labels=list(freq_bounds)[1:])

    # Filter to retain only relevant frequency bands:
    freq_bands_of_interest = ['delta', 'theta', 'alpha', 'beta']
    df = tfr_df[tfr_df.band.isin(freq_bands_of_interest)]
    df['band'] = df['band'].cat.remove_unused_categories()

    # Order channels for plotting:
    df['channel'] = df['channel'].cat.reorder_categories(want_chs, ordered=True)

    g = sns.FacetGrid(df, row='band', col='channel', margin_titles=True)
    g.map(sns.lineplot, 'time', 'value', 'condition', n_boot=10)
    axline_kw = dict(color='black', linestyle='dashed', linewidth=0.5, alpha=0.5)
    g.map(plt.axhline, y=0, **axline_kw)
    g.map(plt.axvline, x=0, **axline_kw)
    g.set(ylim=(vmin, vmax))
    g.set_axis_labels("Time (s)", "ERDS (%)")
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.add_legend(ncol=2, loc='lower center')
    g.fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.08)


def draw_p_test_EpochsTF(tfr, want_chs, event_ids, vmin=-1.5, vmax=6):
    cmap = center_cmap(plt.cm.RdBu, vmin, vmax)  # zero maps to white
    kwargs = dict(n_permutations=100, step_down_p=0.05, seed=1,
                  buffer_size=None, out_type='mask')  # for cluster test
    n_chan = len(want_chs)
    for event in event_ids:
        # select desired epochs for visualization
        tfr_ev = tfr[event]
        fig, axes = plt.subplots(1, n_chan+1)
        for ch, ax in enumerate(axes[:-1]):  # for each channel
            # positive clusters
            _, c1, p1, _ = pcluster_test(tfr_ev.data[:, ch, ...], tail=1, **kwargs)
            # negative clusters
            _, c2, p2, _ = pcluster_test(tfr_ev.data[:, ch, ...], tail=-1, **kwargs)

            if len(c1)==0:
                c = np.stack(c2, axis=2)
                p = p2
            elif len(c2)==0:
                c = np.stack(c1, axis=2)
                p = p1
            else:
                # note that we keep clusters with p <= 0.05 from the combined clusters
                # of two independent tests; in this example, we do not correct for
                # these two comparisons
                c = np.stack(c1 + c2, axis=2) # combined clusters
                p = np.concatenate((p1, p2))  # combined p-values
            mask = c[..., p <= 0.05].any(axis=-1)

            # plot TFR (ERDS map with masking)
            tfr_ev.average().plot([ch], vmin=vmin, vmax=vmax, cmap=(cmap, False),
                                  axes=ax, colorbar=False, show=False, mask=mask,
                                  mask_style="mask")

            ax.set_title(tfr.ch_names[ch], fontsize=10)
            ax.axvline(0, linewidth=1, color="black", linestyle=":")  # event
            if ch != 0:
                ax.set_ylabel("")
                ax.set_yticklabels("")
        fig.colorbar(axes[0].images[-1], cax=axes[-1])
        fig.suptitle("ERDS ({})".format(event))
        fig.show()




