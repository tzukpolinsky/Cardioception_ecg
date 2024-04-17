import os.path
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from metadpy import sdt
from metadpy.plotting import plot_confidence
from metadpy.utils import discreteRatings, trials2counts
from scipy.stats import norm
import neurokit2 as nk


def run_hrd_report(result_path: str,sfreq:int,output_folder:str):
    sns.set_context('talk')
    df = pd.read_csv(
        [file for file in Path(result_path).glob('*final.csv')][0]
    )

    # History of posteriors distribution
    try:
        interoPost = np.load(
            [file for file in Path(result_path).glob('*Intero_posterior.npy')][0]
        )
    except:
        interoPost = None
    try:
        exteroPost = np.load(
            [file for file in Path(result_path).glob('*Extero_posterior.npy')][0]
        )
    except:
        exteroPost = None

    # PPG signal
    signal_df = pd.read_csv(
        [file for file in Path(result_path).glob('*signal.csv')][0]
    )
    signal_df['Time'] = np.arange(0, len(signal_df)) / sfreq  # Create time vector
    palette = ['#b55d60', '#5f9e6e']

    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for i, task, title in zip([0, 1], ['DecisionRT', 'ConfidenceRT'], ['Decision', 'Confidence']):
        sns.boxplot(data=df, x='Modality', y=task, hue='ResponseCorrect',
                    palette=palette, width=.15, notch=True, ax=axs[i])
        sns.stripplot(data=df, x='Modality', y=task, hue='ResponseCorrect',
                      dodge=True, linewidth=1, size=6, palette=palette, alpha=.6, ax=axs[i])
        axs[i].set_title(title)
        axs[i].set_ylabel('Response Time (s)')
        axs[i].set_xlabel('')
        axs[i].get_legend().remove()
    sns.despine(trim=10)

    handles, labels = axs[0].get_legend_handles_labels()
    plt.legend(handles[0:2], ['Incorrect', 'Correct'], bbox_to_anchor=(1.05, .5), loc=2, borderaxespad=0.)
    for i, cond in enumerate(['Intero', 'Extero']):
        this_df = df[df.Modality == cond].copy()
        if len(this_df) > 0:
            this_df['Stimuli'] = (this_df.responseBPM > this_df.listenBPM)
            this_df['Responses'] = (this_df.Decision == 'More')

            hit, miss, fa, cr = this_df.scores()
            if hit+miss == 0:
                continue
            hr, far = sdt.rates(hits=hit, misses=miss, fas=fa, crs=cr)
            d, c = sdt.dprime(hit_rate=hr, fa_rate=far), sdt.criterion(hit_rate=hr, fa_rate=far)

            print(f'Condition: {cond} - d-prime: {d} - criterion: {c}')
    if output_folder != "":
        plt.savefig(os.path.join(output_folder,"Decision Confidence.png"))
    else:
        plt.show()
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))

    for i, cond in enumerate(['Intero', 'Extero']):
        try:
            this_df = df[(df.Modality == cond) & (df.RatingProvided == 1)]
            this_df = this_df[~this_df.Confidence.isnull()]
            new_confidence, _ = discreteRatings(this_df.Confidence)
            this_df['Confidence'] = new_confidence
            this_df['Stimuli'] = (this_df.Alpha > 0).astype('int')
            this_df['Responses'] = (this_df.Decision == 'More').astype('int')
            nR_S1, nR_S2 = trials2counts(data=this_df)
            plot_confidence(nR_S1, nR_S2, ax=axs[i])
            axs[i].set_title(f'{cond}ception')
        except:
            print('Invalid ratings')
            this_df = df[df.Modality == cond]
            sns.histplot(this_df[this_df.ResponseCorrect == 1].Confidence, ax=axs[i], color="#5f9e6e", )
            sns.histplot(this_df[this_df.ResponseCorrect == 0].Confidence, ax=axs[i], color="#b55d60")
            axs[i].set_title(f'{cond}ception')
    sns.despine()
    plt.tight_layout()
    if output_folder != "":
        plt.savefig(os.path.join(output_folder, "Intero Extero.png"))
    else:
        plt.show()
    fig, axs = plt.subplots(1, 1, figsize=(8, 5))

    for cond, col in zip(['Intero', 'Extero'], ['#c44e52', '#4c72b0']):
        this_df = df[df.Modality == cond]
        axs.hist(this_df.Alpha, color=col, bins=np.arange(-40.5, 40.5, 5), histtype='stepfilled',
                 ec="k", density=True, align='mid', label=cond, alpha=.6)
    axs.set_title('Distribution of the tested intensities values')
    axs.set_xlabel('Intensity (BPM)')
    plt.legend()
    sns.despine(trim=10)
    plt.tight_layout()
    if sum(df.TrialType == 'psi') > 0:

        fig, axs = plt.subplots(figsize=(18, 5), nrows=1, ncols=2)

        # Plot confidence interval for each staircase
        def ci(x):
            return np.where(np.cumsum(x) / np.sum(x) > .025)[0][0], \
                np.where(np.cumsum(x) / np.sum(x) < .975)[0][-1]

        try:
            for i, stair, col, modality in zip([0, 1],
                                               [interoPost, exteroPost],
                                               ['#c44e52', '#4c72b0'],
                                               ['Intero', 'Extero']):
                this_df = df[(df.Modality == modality) & (df.TrialType != 'UpDown')]
                ciUp, ciLow = [], []
                for t in range(stair.shape[0]):
                    up, low = ci(stair.mean(2)[t])
                    rg = np.arange(-50.5, 50.5)
                    ciUp.append(rg[up])
                    ciLow.append(rg[low])

                axs[i].fill_between(x=np.linspace(0, len(this_df), len(ciUp)),
                                    y1=ciLow,
                                    y2=ciUp,
                                    color=col, alpha=.2)
        except:
            pass

        # Staircase traces
        for i, modality, col in zip([0, 1], ['Intero', 'Extero'], ['#c44e52', '#4c72b0']):
            this_df = df[(df.Modality == modality) & (df.TrialType != 'UpDown')]

            # Show UpDown staircase traces
            axs[i].plot(np.arange(0, len(this_df))[this_df.TrialType == 'high'],
                        this_df.Alpha[this_df.TrialType == 'high'], linestyle='--', color=col, linewidth=2)
            axs[i].plot(np.arange(0, len(this_df))[this_df.TrialType == 'low'],
                        this_df.Alpha[this_df.TrialType == 'low'], linestyle='-', color=col, linewidth=2)

            # Use different colors for psi and catch trials
            for trialCond, pointCol in zip(['psi', 'psiCatchTrial'], [col, 'gray']):
                axs[i].plot(np.arange(0, len(this_df))[(this_df.Decision == 'More') & (this_df.TrialType == trialCond)],
                            this_df.Alpha[(this_df.Decision == 'More') & (this_df.TrialType == trialCond)],
                            pointCol, marker='o', linestyle='', markeredgecolor='k', label=cond)
                axs[i].plot(np.arange(0, len(this_df))[(this_df.Decision == 'Less') & (this_df.TrialType == trialCond)],
                            this_df.Alpha[(this_df.Decision == 'Less') & (this_df.TrialType == trialCond)],
                            'w', marker='s', linestyle='', markeredgecolor=pointCol, label=modality)

            # Psi trials
            axs[i].plot(np.arange(len(this_df))[this_df.TrialType == 'psi'],
                        this_df[this_df.TrialType == 'psi'].EstimatedThreshold, linestyle='-', color=col, linewidth=4)

            axs[i].axhline(y=0, linestyle='--', color='gray')
            handles, labels = axs[i].get_legend_handles_labels()
            axs[i].legend(handles[0:2], ['More', 'Less'], borderaxespad=0., title='Decision')
            axs[i].set_ylabel('Intensity ($\Delta$ BPM)')
            axs[i].set_xlabel('Trials')
            axs[i].set_ylim(-52, 52)
            axs[i].set_title(modality + 'ception')
            sns.despine(trim=10, ax=axs[i])
            plt.gcf()
    if output_folder != "":
        plt.savefig(os.path.join(output_folder,"Distribution of the tested intensities values.png"))
    else:
        plt.show()
    sns.set_context('talk')
    fig, axs = plt.subplots(figsize=(8, 5))
    for i, modality, col in zip((0, 1), ['Extero', 'Intero'], ['#4c72b0', '#c44e52']):

        this_df = df[(df.Modality == modality) & (df.TrialType == 'psi')]
        if len(this_df) > 0:
            t, s = this_df.EstimatedThreshold.iloc[-1], this_df.EstimatedSlope.iloc[-1]
            # Plot Psi estimate of psychometric function
            axs.plot(np.linspace(-40, 40, 500),
                     (norm.cdf(np.linspace(-40, 40, 500), loc=t, scale=s)),
                     '--', color=col, label=modality)
            # Plot threshold
            axs.plot([t, t], [0, .5], color=col, linewidth=2)
            axs.plot(t, .5, 'o', color=col, markersize=10)

            # Plot data points
            for ii, intensity in enumerate(np.sort(this_df.Alpha.unique())):
                resp = sum((this_df.Alpha == intensity) & (this_df.Decision == 'More'))
                total = sum(this_df.Alpha == intensity)
                axs.plot(intensity, resp / total, 'o', alpha=0.5, color=col,
                         markeredgecolor='k', markersize=total * 5)
    plt.ylabel('P$_{(Response = More|Intensity)}$')
    plt.xlabel('Intensity ($\Delta$ BPM)')
    plt.tight_layout()
    plt.legend()
    sns.despine()
    if output_folder != "":
        plt.savefig(os.path.join(output_folder, "Intensity Response.png"))
    else:
        plt.show()
    drop, bpm_std, bpm_df = [], [], pd.DataFrame([])
    clean_df = df.copy()
    clean_df['HeartRateOutlier'] = np.zeros(len(clean_df), dtype='bool')
    for i, trial in enumerate(signal_df.nTrial.unique()):
        color = '#c44e52' if (i % 2) == 0 else '#4c72b0'
        this_df = signal_df[signal_df.nTrial == trial]  # Downsample to save memory

        cleaned = nk.ecg_clean(this_df.signal, sampling_rate=sfreq, method="neurokit")

        signals, info = nk.ecg_peaks(cleaned, sampling_rate=sfreq, method="neurokit")
        peaks = np.array(signals["ECG_R_Peaks"]).astype(bool)
        bpm = sfreq*60 / np.diff(np.where(peaks)[0])

        bpm_df = pd.concat(
            [
                bpm_df,
                pd.DataFrame({'bpm': bpm, 'nEpoch': i, 'nTrial': trial})
            ]
        )

    # Check for outliers in the absolute value of RR intervals
    for e, t in zip(bpm_df.nEpoch[pg.madmedianrule(bpm_df.bpm.to_numpy())].unique(),
                    bpm_df.nTrial[pg.madmedianrule(bpm_df.bpm.to_numpy())].unique()):
        drop.append(e)
        clean_df.loc[t, 'HeartRateOutlier'] = True

    # Check for outliers in the standard deviation values of RR intervals
    for e, t in zip(np.arange(0, bpm_df.nTrial.nunique())[
                        pg.madmedianrule(bpm_df.copy().groupby(['nTrial', 'nEpoch']).bpm.std().to_numpy())],
                    bpm_df.nTrial.unique()[
                        pg.madmedianrule(bpm_df.copy().groupby(['nTrial', 'nEpoch']).bpm.std().to_numpy())]):
        if e not in drop:
            drop.append(e)
            clean_df.loc[t, 'HeartRateOutlier'] = True
    meanBPM, stdBPM, rangeBPM = [], [], []

    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(30, 10))
    for i, trial in enumerate(signal_df.nTrial.unique()):

        color = '#3a5799' if (i % 2) == 0 else '#3bb0ac'
        this_df = signal_df[signal_df.nTrial == trial]  # Downsample to save memory

        # Mark as outlier if relevant
        if i in drop:
            ax[0].axvspan(this_df.Time.iloc[0], this_df.Time.iloc[-1], alpha=.3, color='gray')
            ax[1].axvspan(this_df.Time.iloc[0], this_df.Time.iloc[-1], alpha=.3, color='gray')

        ax[0].plot(this_df.Time, this_df.signal, label='PPG', color=color, linewidth=.5)
        cleaned = nk.ecg_clean(this_df.signal, sampling_rate=sfreq, method="neurokit")

        signals, info = nk.ecg_peaks(cleaned, sampling_rate=sfreq, method="neurokit")
        peaks = np.array(signals["ECG_R_Peaks"]).astype(bool)
        # Peaks detection
        bpm = sfreq*60 / np.diff(np.where(peaks)[0])
        m, s, r = bpm.mean(), bpm.std(), bpm.max() - bpm.min()
        meanBPM.append(m)
        stdBPM.append(s)
        rangeBPM.append(r)

        # Plot instantaneous heart rate
        ax[1].plot(this_df.Time.to_numpy()[np.where(peaks)[0][1:]],
                   sfreq*60 / np.diff(np.where(peaks)[0]),
                   'o-', color=color, alpha=0.6)

    ax[1].set_xlabel("Time (s)")
    ax[0].set_ylabel("PPG level (a.u.)")
    ax[1].set_ylabel("Heart rate (BPM)")
    ax[0].set_title("PPG signal recorded during interoceptive condition (5 seconds each)")
    sns.despine()
    ax[0].grid(True)
    ax[1].grid(True)
    if output_folder != "":
        plt.savefig(os.path.join(output_folder, "signal recorded during interoceptive condition (5 seconds each).png"))
    else:
        plt.show()
    sns.set_context('talk')
    fig, axs = plt.subplots(figsize=(13, 5), nrows=2, ncols=2)
    meanBPM = np.delete(np.array(meanBPM), np.array(drop))
    stdBPM = np.delete(np.array(stdBPM), np.array(drop))
    for i, metric, col in zip(range(3), [meanBPM, stdBPM], ['#b55d60', '#5f9e6e']):
        axs[i, 0].plot(metric, 'o-', color=col, alpha=.6)
        axs[i, 1].hist(metric, color=col, bins=15, ec="k", density=True, alpha=.6)
        axs[i, 0].set_ylabel('Mean BPM' if i == 0 else 'STD BPM')
        axs[i, 0].set_xlabel('Trials')
        axs[i, 1].set_xlabel('BPM')
    sns.despine()
    plt.tight_layout()
    print(
        f'{clean_df["HeartRateOutlier"][clean_df.Modality == "Intero"].sum()} Interoception trials and {clean_df["HeartRateOutlier"][clean_df.Modality == "Extero"].sum()} exteroception trials were dropped after trial rejection based on heart rate outliers.')
    if output_folder != "":
        plt.savefig(os.path.join(output_folder, "Trials analysis.png"))
    else:
        plt.show()


if __name__ == '__main__':
    run_hrd_report("C:\\Users\\rsolomon\\data\\safeHeart\\intreospection\\results\\",250,"C:\\Users\\rsolomon\\data\\safeHeart\\intreospection\\results\\")