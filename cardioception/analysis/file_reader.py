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


def read_trials(file_name: str) -> pd.DataFrame:
    df = None
    headers_cols = ['TrialType', 'Condition', 'Modality', 'StairCond', 'Decision', 'DecisionRT', 'Confidence',
                    'ConfidenceRT', 'Alpha', 'listenBPM', 'responseBPM', 'ResponseCorrect', 'DecisionProvided',
                    'RatingProvided', 'nTrials', 'EstimatedThreshold', 'EstimatedSlope', 'StartListening',
                    'StartDecision', 'ResponseMade', 'RatingStart', 'RatingEnds', 'endTrigger']
    try:
        df = pd.read_csv(file_name)
        cols = df.columns.to_list()
        if len(cols) != len(headers_cols):
            print(f'the trails file is not with the expected header')
        for c in cols:
            if c not in headers_cols:
                print(f'the trails file has an unexpected column name: {c}')
        for h in headers_cols:
            if h not in cols:
                print(f'the trails file is missing an expected column name: {h}')
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
