import pandas as pd


def read_signal(file_name: str) -> pd.DataFrame:
    df = None
    try:
        df = pd.read_csv(file_name)
        cols = df.columns.to_list()
        if len(cols) != 3 or ("signal" in cols and 'nTrial' in cols and 'time' not in cols):
            print(f'the signal file is not with the expected header')
    except Exception as e:
        print("couldn't read csv file from the following reason")
        print(f'{e.args[1]}')
    return df


def read_parameters(file_name: str) -> pd.DataFrame:
    df = None
    try:
        df = pd.read_pickle(file_name)
        cols = df.columns.to_list()
    except Exception as e:
        print("couldn't read csv file from the following reason")
        print(f'{e.args[1]}')
    return df
