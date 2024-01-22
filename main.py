import os
from datetime import datetime
from cardioception.reports import report
from cardioception.HRD.parameters import getParameters
from cardioception.HRD import task

if __name__ == "__main__":
    now = datetime.now().strftime('%Y-%m-%d%H-%M-%S')
    results_path = os.path.join(os.getcwd(), os.path.join("results", f"{now}.txt"))
    # Set global task parameters
    parameters = getParameters(
        participant='Subject_01', session='Test', serialPort=None, data_stream_device='zephyr', samples_per_second=250,
        setup='test', nTrials=2, screenNb=0, resultPath=results_path)

    # Run task
    task.run(parameters, confidenceRating=True, runTutorial=False)

    parameters['win'].close()
    reports_path = os.path.join(os.getcwd(), os.path.join("resports", f"{now}.html"))
    report(results_path, reports_path, task='HRD')
