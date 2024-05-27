import os
from datetime import datetime
from pathlib import Path
from psychopy import core
from cardioception.reports import report
from cardioception.HRD.parameters import getParameters
from cardioception.HRD import task
from cardioception.HRD.HRDReport import run_hrd_report

if __name__ == "__main__":
    now = datetime.now().strftime('%Y-%m-%d%H-%M-%S')
    results_path = os.path.join(os.getcwd(), os.path.join("results", f"{now}"))
    # Set global task parameters
    participant_name = 'Subject_01'
    session_name = 'Test'
    samples_per_seconds = 250
    parameters = getParameters(language='hebrew',
                               participant='Subject_01', session='Test', serialPort=None, fullscr=False,
                               exteroception=True, data_stream_device='zephyr', samples_per_second=samples_per_seconds,
                               setup='behavioral', nTrials=10, screenNb=0, device='keyboard', resultPath=results_path,
                               address='58:93:D8:4A:6A:08', maxRatingTime=10, respMax=10, nFeedback=2, nConfidence=2)
    # Run task
    task.run(parameters, confidenceRating=True, runTutorial=True)

    # run_hrd_report(results_path,samples_per_seconds,results_path)
