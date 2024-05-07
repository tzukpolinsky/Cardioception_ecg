import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde, ttest_ind, ttest_1samp, t, sem, chi2_contingency
import ptitprince as pt
import neurokit2 as nk


def plot_signal(heart_signal_data: pd.DataFrame):
    plt.figure(figsize=(10, 5))
    plt.plot(heart_signal_data['time'], heart_signal_data['signal'], label='Heart Signal')
    plt.xlabel('Time')
    plt.ylabel('Signal Amplitude')
    plt.title('Heart Signal Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()


def get_signal_basic_statistics(heart_signal_data: pd.DataFrame):
    signal_stats = {
        "Mean": heart_signal_data['signal'].mean(),
        "Median": heart_signal_data['signal'].median(),
        "Standard Deviation": heart_signal_data['signal'].std(),
        "Minimum": heart_signal_data['signal'].min(),
        "Maximum": heart_signal_data['signal'].max(),
        "Range": heart_signal_data['signal'].max() - heart_signal_data['signal'].min()
    }

    return signal_stats


def plot_rpeak_distances_per_trail(heart_signal_data: pd.DataFrame, sampling_rate=250):
    grouped_stats = heart_signal_data.groupby('nTrial')
    rpeak_df_dict = {'rpeak_distances': [], 'nTrial': []}
    for trail_number, trail_data in grouped_stats:
        signal = trail_data['signal']
        cleaned = nk.ecg_clean(signal, sampling_rate=sampling_rate, method="neurokit")
        if len(cleaned) < sampling_rate * 0.1:
            print("empty cleaned signal")
            return [], np.array([])
        peaks_indices, info = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate, method="neurokit")
        peaks = (sampling_rate * 60) / np.diff(np.where(peaks_indices))[0]
        rpeak_df_dict['rpeak_distances'].append(np.std(peaks))
        rpeak_df_dict['nTrial'].append(trail_number)
        # for i in range(1, len(peaks)):
        #     rpeak_df_dict['rpeak_distances'].append(peaks[i] - peaks[i - 1])
        #     rpeak_df_dict['nTrial'].append(trail_number)
    df = pd.DataFrame(rpeak_df_dict)
    ax = sns.stripplot(x='nTrial', y='rpeak_distances', data=df, dodge=True, edgecolor="white",
                       size=8, jitter=1, zorder=0, orient="v", alpha=0.8, clip_on=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('rpeak distances', labelpad=10, fontsize=25)
    plt.tight_layout(pad=2.0)
    plt.title('rpeak distances per trial number')
    plt.show()


def get_trails_basic_statistics(heart_signal_data: pd.DataFrame):
    grouped_stats = heart_signal_data.groupby('nTrial')['signal'].agg(['mean', 'median', 'std', 'min', 'max'])

    # Calculate the range for each group
    grouped_stats['range'] = grouped_stats['max'] - grouped_stats['min']

    # Display grouped statistics and overall statistics for reference
    overall_stats = {
        "Overall Mean": heart_signal_data['signal'].mean(),
        "Overall Median": heart_signal_data['signal'].median(),
        "Overall Std Deviation": heart_signal_data['signal'].std(),
        "Overall Min": heart_signal_data['signal'].min(),
        "Overall Max": heart_signal_data['signal'].max(),
        "Overall Range": heart_signal_data['signal'].max() - heart_signal_data['signal'].min()
    }

    return grouped_stats, overall_stats


def plot_trails_basic_statistics(heart_signal_data: pd.DataFrame):
    grouped_stats, overall_stats = get_trails_basic_statistics(heart_signal_data)
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Plot Mean Signal Value per Trial
    axes[0].bar(grouped_stats.index, grouped_stats['mean'], color='blue')
    axes[0].set_title('Mean Signal Value per Trial')
    axes[0].set_xlabel('Trial Number')
    axes[0].set_ylabel('Mean Signal Value')

    # Plot Standard Deviation per Trial
    axes[1].bar(grouped_stats.index, grouped_stats['std'], color='green')
    axes[1].set_title('Standard Deviation of Signal per Trial')
    axes[1].set_xlabel('Trial Number')
    axes[1].set_ylabel('Standard Deviation')

    # Plot Range of Signal per Trial
    axes[2].bar(grouped_stats.index, grouped_stats['range'], color='red')
    axes[2].set_title('Range of Signal per Trial')
    axes[2].set_xlabel('Trial Number')
    axes[2].set_ylabel('Range (Max - Min)')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()


def plot_trials_column_basic_statistics(trails_data: pd.DataFrame, col_name: str):
    overall_stats = get_trails_column_basic_statistics(trails_data, col_name)
    raincloud_plot(trails_data, col_name, f'trails {col_name}')
    # plt.figure(figsize=(14, 6))
    # plt.bar(trails_data['nTrail'], trails_data[col_name], color='purple')
    # plt.title(f'{col_name} of Each Trial')
    # plt.xlabel('Trial Number')
    # plt.ylabel(f'{col_name}')
    # plt.show()
    # return trial_times, duration_stats


def get_trails_column_basic_statistics(trails_signal_data: pd.DataFrame, col_name: str):
    # Display grouped statistics and overall statistics for reference
    stats = {
        f"{col_name} overall Mean": trails_signal_data[col_name].mean(),
        f"{col_name} overall Median": trails_signal_data[col_name].median(),
        f"{col_name} overall Std Deviation": trails_signal_data[col_name].std(),
        f"{col_name} overall Min": trails_signal_data[col_name].min(),
        f"{col_name} overall Max": trails_signal_data[col_name].max(),
        f"{col_name} overall Range": trails_signal_data[col_name].max() - trails_signal_data[col_name].min()
    }

    return stats


def get_trails_times_basic_statistics(heart_signal_data: pd.DataFrame):
    time_stats = {
        "Mean": heart_signal_data['time'].mean(),
        "Median": heart_signal_data['time'].median(),
        "Standard Deviation": heart_signal_data['time'].std(),
        "Minimum": heart_signal_data['time'].min(),
        "Maximum": heart_signal_data['time'].max(),
        "Range": heart_signal_data['time'].max() - heart_signal_data['time'].min()
    }

    return time_stats


def raincloud_plot(df: pd.DataFrame, dy: str, title: str):
    ort = "v"
    ax = pt.half_violinplot(y=dy, data=df, bw=.2, cut=0.,
                            scale="area", width=.6, inner=None, orient=ort)
    ax = sns.stripplot(y=dy, data=df, dodge=True, edgecolor="white",
                       size=8, jitter=1, zorder=0, orient=ort, alpha=0.1, clip_on=False)
    ax = sns.pointplot(y=dy, data=df, dodge=True, join=False,
                       markers='D', capsize=0.2, errwidth=0.9, ci=None, scale=1.7, clip_on=False, orient=ort)
    mean = df[dy].mean()
    s = sem(df[dy].to_numpy())
    ci = t.interval(0.95, len(df) - 1, loc=mean, scale=s)
    plt.errorbar(
        x=1,
        y=mean,
        yerr=np.array([mean - ci[0], ci[1] - mean]).reshape(-1, 1),
        fmt='none',
        capsize=5,
        elinewidth=1.2,
        capthick=1.2

    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel(dy, labelpad=10, fontsize=25)
    plt.tight_layout(pad=2.0)
    plt.title(title)
    plt.show()
