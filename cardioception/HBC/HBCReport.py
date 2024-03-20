from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import numpy as np
import pandas as pd
import seaborn as sns
from systole.detection import ppg_peaks
from systole.plots import plot_raw, plot_subspaces


def run_hbc_report():
    sns.set_context('paper')
    resultPath = Path(Path.cwd(), "data", "HBC")
    reportPath = Path(Path.cwd(), "reports")
    results_df = [file for file in Path(resultPath).glob('*final.txt')]
    df = pd.read_csv(results_df[0])
    ppg = {}
    for i in range(6):
        ppg[str(i)] = np.load(
            [file for file in resultPath.glob(f'*_{i}.npy')][0]
        )
    counts = []
    for nTrial in range(6):

        print(f'Analyzing trial number {nTrial + 1}')

        signal, peaks = ppg_peaks(ppg[str(nTrial)][0], clean_extra=True, sfreq=75)
        axs = plot_raw(
            signal=signal, sfreq=1000, figsize=(18, 5), clean_extra=True,
            show_heart_rate=True
        )

        # Show the windows of interest
        # We need to convert sample vector into Matplotlib internal representation
        # so we can index it easily
        x_vec = date2num(
            pd.to_datetime(
                np.arange(0, len(signal)), unit="ms", origin="unix"
            )
        )
        l = len(signal) / 1000
        for i in range(2):
            # Pre-trial time
            axs[i].axvspan(
                x_vec[0], x_vec[- (3 + df.Duration.iloc[nTrial]) * 1000]
                , alpha=.2
            )
            # Post trial time
            axs[i].axvspan(
                x_vec[- 3 * 1000],
                x_vec[- 1],
                alpha=.2
            )
        plt.show()

        # Detected heartbeat in the time window of interest
        peaks = peaks[int(l - (3 + df.Duration.iloc[nTrial])) * 1000:int((l - 3) * 1000)]

        rr = np.diff(np.where(peaks)[0])

        _, axs = plt.subplots(ncols=2, figsize=(12, 6))
        plot_subspaces(rr=rr, ax=axs)
        plt.show()

        trial_counts = np.sum(peaks)
        print(f'Reported: {df.Reported.loc[nTrial]} beats ; Detected : {trial_counts} beats')
        counts.append(trial_counts)
    df['Counts'] = counts
    df['Score'] = 1 - ((df.Counts - df.Reported).abs() / ((df.Counts + df.Reported) / 2))
    df.to_csv(Path(resultPath, 'processed.txt'))
