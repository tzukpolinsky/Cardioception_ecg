import sys

from cardioception.analysis.file_reader import read_signal, read_trials
from cardioception.analysis.plots.simple_statistics import *
from cardioception.analysis.plots.continuous_plots import *


def main():
    signal_file_path = sys.argv[1]
    trails_file_path = sys.argv[2]
    trails_df = read_trials(trails_file_path)
    heart_signal_df = read_signal(signal_file_path)
    plot_rpeak_distances_per_trail(heart_signal_df)
    plot_trails_basic_statistics(heart_signal_df)
    plot_trials_column_basic_statistics(trails_df, 'Alpha')
    plot_column_convergence_over_nTrials(trails_df, 'Alpha')


if __name__ == '__main__':
    main()
