import os
from datetime import datetime
from pathlib import Path

from cardioception.reports import report
from cardioception.HRD.parameters import getParameters
from cardioception.HRD import task
from cardioception.HRD.HRDReport import run_hrd_report

if __name__ == "__main__":
    now = datetime.now().strftime('%Y-%m-%d%H-%M-%S')
    results_path = os.path.join(os.getcwd(), os.path.join("results", f"{now}.txt"))
    # Set global task parameters
    participant_name = 'Subject_01'
    session_name = 'Test'
    samples_per_seconds = 250
    parameters = getParameters(language='hebrew',
        participant='Subject_01', session='Test', serialPort=None,fullscr=False,exteroception=False, data_stream_device='zephyr', samples_per_second=samples_per_seconds,
        setup='behavioral', nTrials=40, screenNb=0, resultPath=results_path,address='58:93:D8:4A:6A:08',maxRatingTime=10,respMax=10)
    # Run task
    task.run(parameters, confidenceRating=True, runTutorial=False)

    parameters['win'].close()
    reports_path = os.path.join(os.getcwd(), os.path.join("resports", f"{now}.html"))
    # report(results_path, reports_path, task='HRD')

    run_hrd_report(results_path,samples_per_seconds)
