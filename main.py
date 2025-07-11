import os
from datetime import datetime
from pathlib import Path
from psychopy import core, gui, data
from cardioception.reports import report
from cardioception.HRD.parameters import getParameters as HRD_getParameters
from cardioception.HBC.parameters import getParameters as HBC_getParameters
from cardioception.HRD import task as HRD_task
from cardioception.HBC import task as HBC_task

# from cardioception.HRD.HRDReport import run_hrd_report

if __name__ == "__main__":
    subject_info = {
        'Subject Number': '',
        'Session': '001',
        'date': data.getDateStr(),
        'language': ['hebrew', 'english'],
        "save folder": os.path.join(os.getcwd(), 'data'),
        "full screen": True,
        "assignment type": ['HRD', 'HBC', 'C-TCT'],
        "screen number": 0,
    }

    # Create a dialog box
    dlg = gui.DlgFromDict(dictionary=subject_info, title='Subject Information')

    # If the user presses 'Cancel', exit the program
    if not dlg.OK:
        core.quit()
    assignment_type = subject_info['assignment type']
    subject = subject_info['Subject Number']
    results_path = os.path.join(os.getcwd(), os.path.join("results", f"{assignment_type}_{subject}"))
    # Set global task parameters
    participant_name = subject_info['Subject Number']
    session = subject_info['Session']

    if assignment_type == 'HRD':
        HRD_subject_info = {
            'number of trials': 100,
            'number of feedback trials': 3,
            'number of confidence trials': 2,
            'exteroception': True,
            'recording device': 'zephyr',
            'device bluetooth address': ['A0:E6:F8:FA:98:7A', '58:93:D8:4A:6A:08', 'A4:DA:32:81:AF:A0'],
            'samples per second': 250,
        }

        # Create a dialog box
        dlg = gui.DlgFromDict(dictionary=HRD_subject_info, title='HRD task parameters')

        # If the user presses 'Cancel', exit the program
        if not dlg.OK:
            core.quit()
        parameters = HRD_getParameters(language=subject_info['language'],
                                       participant=participant_name, session=session, serialPort=None,
                                       fullscr=subject_info['full screen'],
                                       exteroception=HRD_subject_info['exteroception'],
                                       data_stream_device=HRD_subject_info['recording device'],
                                       samples_per_second=HRD_subject_info['samples per second'],
                                       setup='behavioral', nTrials=HRD_subject_info['number of trials'],
                                       screenNb=subject_info['screen number'],
                                       device='keyboard', resultPath=results_path,
                                       address=HRD_subject_info['device bluetooth address'], maxRatingTime=10,
                                       respMax=10,
                                       nFeedback=HRD_subject_info['number of feedback trials'],
                                       nConfidence=HRD_subject_info['number of confidence trials'])
        # Run task
        if HRD_task.run(parameters, confidenceRating=True, runTutorial=True):
            print('user aborted the task in the middle')

    elif assignment_type == 'HBC':
        HBC_subject_info = {
            'task version': ['Zaccaro', 'Garfinkel', 'Schandry', 'test'],
            'exteroception': False,
            'data_stream_device': ['EEG'],
            'EEG triggers port': '0x6EFC',
            'EEG trigger pulse (MS)': 4,
            'task setup': ['behavioral', 'test']
        }

        # Create a dialog box
        dlg = gui.DlgFromDict(dictionary=HBC_subject_info, title='HBC Task parameters')

        # If the user presses 'Cancel', exit the program
        if not dlg.OK:
            core.quit()
        parameters = HBC_getParameters(language=subject_info['language'],
                                       participant=participant_name, session=session,
                                       EEG_trigger_pulse=HBC_subject_info['EEG trigger pulse (MS)'],
                                       EEG_triggers_port=int(HBC_subject_info['EEG triggers port'], 16),
                                       taskVersion=HBC_subject_info['task version'],
                                       serialPort='', systole_kw={},
                                       fullscr=subject_info['full screen'],
                                       exteroception=HBC_subject_info['exteroception'],
                                       data_stream_device='EEG',
                                       setup='behavioral', maxRatingTime=10,
                                       screenNb=subject_info['screen number'], resultPath=results_path)
        # Run task
        if HBC_task.run(parameters, runTutorial=True):
            print('user aborted the task in the middle')


    elif assignment_type == 'C-TCT':
        HBC_subject_info = {
            'task version': ['Zaccaro', 'Garfinkel', 'Schandry', 'test'],
            'exteroception': True,
            'data_stream_device': ['EEG'],
            'EEG triggers port': '0x6EFC',
            'EEG trigger pulse (MS)': 4,
            'task setup': ['behavioral', 'test']
        }

        # Create a dialog box
        dlg = gui.DlgFromDict(dictionary=HBC_subject_info, title='C-TCT Task parameters')

        # If the user presses 'Cancel', exit the program
        if not dlg.OK:
            core.quit()
        parameters = HBC_getParameters(language=subject_info['language'],
                                       participant=participant_name, session=session,
                                       EEG_trigger_pulse=HBC_subject_info['EEG trigger pulse (MS)'],
                                       EEG_triggers_port=int(HBC_subject_info['EEG triggers port'], 16),
                                       taskVersion=HBC_subject_info['task version'],
                                       serialPort='', systole_kw={},
                                       fullscr=subject_info['full screen'],
                                       exteroception=HBC_subject_info['exteroception'],
                                       data_stream_device='EEG',
                                       setup='behavioral', maxRatingTime=10,
                                       screenNb=subject_info['screen number'], resultPath=results_path)
        # Run task
        if HBC_task.run(parameters, runTutorial=True):
            print('user aborted the task in the middle')

#    run_hrd_report(results_path,samples_per_seconds,results_path)
