import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde, ttest_ind, ttest_1samp, t, sem, chi2_contingency
import statsmodels.api as sm


def plot_column_convergence_over_nTrials(trails_data: pd.DataFrame, col_name: str):
    plt.figure(figsize=(10, 6))
    plt.plot(trails_data['nTrials'], trails_data[col_name], marker='o', linestyle='-', color='blue')
    trails_data['moving_average'] = trails_data[col_name].rolling(window=10).mean()
    plt.plot(trails_data['nTrials'], trails_data['moving_average'], label='Moving Average', color='red')

    # Fitting a linear regression to predict the trend
    X = sm.add_constant(trails_data['nTrials'])  # adding a constant for the intercept
    model = sm.OLS(trails_data[col_name], X).fit()
    predictions = model.predict(X)  # make the predictions by the model

    plt.plot(trails_data['nTrials'], predictions, label='Fitted Line', color='green')

    plt.title(f'Convergence of {col_name} over Trials')
    plt.xlabel('Trial Number')
    plt.ylabel(f'{col_name}')
    plt.legend()
    plt.grid(True)
    plt.show()
