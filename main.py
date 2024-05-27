import os
from datetime import datetime
from pathlib import Path
from psychopy import core, gui, data
from cardioception.reports import report
from cardioception.HRD.parameters import getParameters
from cardioception.HRD import task
from cardioception.HRD.HRDReport import run_hrd_report

if __name__ == "__main__":
    subject_info = {
        'Subject Number': 'Tzuk',
        'Gender': ['Male', 'Female', 'Other'],
        'Session': '001',
        'exteroception': True,
        'number of trials': 50,
        'number of feedback trials': 2,
        'number of confidence trials': 2,
        'user device': ['keyboard', 'mouse'],
        'recording device': 'zephyr',
        'device bluetooth address': ['58:93:D8:4A:6A:08'],
        'date': data.getDateStr(),
        'samples per seconds': 250,
        'language': ['hebrew', 'english'],
        "save folder": os.path.join(os.getcwd(), 'data'),
        "full screen": True,
        "screen number": 0,
    }

    # Create a dialog box
    dlg = gui.DlgFromDict(dictionary=subject_info, title='Subject Information')

    # If the user presses 'Cancel', exit the program
    if not dlg.OK:
        core.quit()
    date = subject_info['date']
    results_path = os.path.join(os.getcwd(), os.path.join("results", f"{date}"))
    # Set global task parameters
    participant_name = subject_info['Subject Number']
    session = subject_info['Session']
    device = subject_info['device']
    parameters = getParameters(language=subject_info['language'],
                               participant=participant_name, session=session, serialPort=None,
                               fullscr=subject_info['full screen'],
                               exteroception=subject_info['exteroception'],
                               data_stream_device=subject_info['recording device'],
                               samples_per_second=subject_info['samples per second'],
                               setup='behavioral', nTrials=subject_info['number of trials'],
                               screenNb=subject_info['screen number'],
                               device=subject_info['user device'], resultPath=results_path,
                               address=subject_info['device bluetooth address'], maxRatingTime=10, respMax=10,
                               nFeedback=subject_info['number of feedback trials'],
                               nConfidence=subject_info['number of confidence trials'])
    # Run task
    task.run(parameters, confidenceRating=True, runTutorial=True)

    # run_hrd_report(results_path,samples_per_seconds,results_path)
