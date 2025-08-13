# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
import random
import time
from typing import Optional, Tuple
import winsound
import numpy as np
import pandas as pd
from psychopy import visual, core, event
from psychopy.hardware import keyboard
import os
import numpy as np
import re
import simpleaudio as sa
import ctypes
from pydub import AudioSegment
import platform
import pkg_resources  # type: ignore
import json
from cardioception.HBC.Sounds.heart_sounds.heartbeat_bpm import repeat_tone_for_duration
import cardioception.HBC.Randomization_CTCT as rnd

def run(
        parameters: dict,
        runTutorial: bool = True,
) -> bool:
    """Run the entire task sequence.

    Parameters
    ----------
    parameters : dict
        Task parameters.
    runTutorial : bool
        If `True`, will present a tutorial with 10 training trial with feedback and 5
        trials with confidence rating.
    Returns
    ---------
    bool: did the subject ended the task
    """

    # Make sure numLock is "on".
    if platform.system() == 'Windows':
        ensure_numlock_on()

    # Run tutorial
    if runTutorial is True:
        parameters['triggers']['tutorialStart']()
        tutorial(parameters)
        parameters['triggers']['tutorialEnd']()
        core.wait(0.5)

    # Randomize the tasks
    if parameters['CTCT']:
        parameters['counterbalance'] = np.random.permutation([True, False]).tolist()
    else:
        parameters['counterbalance'] = [False]

    tasks = pd.DataFrame()

    for idx, condition in enumerate(parameters['counterbalance']):
        parameters["exteroception"] = condition
        if parameters["exteroception"] == False:
            task_start_time = core.getTime()
            parameters['triggers']['HBC_Start']()

            task = "HBC"

            messageStart = visual.TextStim(
                parameters["win"],
                height=parameters["textSize"],
                text=parameters["texts"]["HBC_Start"],
                languageStyle=parameters['languageStyle'],
                wrapWidth=15
            )
            messageStart.draw()
            press = visual.TextStim(
                parameters["win"],
                height=parameters["textSize"],
                text=parameters["texts"]["continue_text"],
                pos=(0.0, -0.4),
                languageStyle=parameters['languageStyle'],
                wrapWidth=15
            )
            press.draw()
            parameters["win"].flip()
            event.waitKeys(keyList=parameters["startKey"])

        else:
            task_start_time = core.getTime()
            parameters['triggers']['CTCT_Start']()

            task = "CTCT"

            messageStart = visual.TextStim(
                parameters["win"],
                height=parameters["textSize"],
                text=parameters["texts"]["CTCT_Start"],
                languageStyle=parameters['languageStyle'],
                wrapWidth=15
            )
            messageStart.draw()
            press = visual.TextStim(
                parameters["win"],
                height=parameters["textSize"],
                text=parameters["texts"]["continue_text"],
                pos=(0.0, -0.4),
                languageStyle=parameters['languageStyle'],
                wrapWidth=15
            )
            press.draw()
            parameters["win"].flip()
            event.waitKeys(keyList=parameters["startKey"])

            bpms = [50, 60, 70]
            part_raw = str(parameters.get("participant", ""))
            seed_val = int(part_raw) if part_raw.isdigit() else None

            durations, bpms = rnd.get_trial_sequence(parameters["times"],bpms, seed=seed_val)
            parameters["bpms_seq"] = np.array(bpms)
            parameters["times"] = np.array(durations)

        if "bpms_seq" not in parameters:
            parameters["bpms_seq"] = np.array([None] * len(parameters["times"]))

        assert len(parameters["bpms_seq"]) == len(parameters["times"]), \
            "bpms_seq length mismatch"
        # Rest
        if parameters["restPeriod"] is True:
            rest(parameters, duration=parameters["restLength"])

        user_aborted = False

        # for condition, duration, nTrial in zip(
        #         parameters["conditions"],
        #         parameters["times"],
        #         range(0, len(parameters["conditions"])),
        # ):
        for nTrial, (condition, duration, bpm) in enumerate(
                zip(parameters["conditions"],
                    parameters["times"],
                    parameters["bpms_seq"])
        ):

            core.wait(0.5)
            parameters['triggers']['trialStart']()

            if parameters["exteroception"] == False:
                nCount, confidence, confidenceRT, actual_duration, user_aborted, bpm, num_beats, modality = trial(condition, duration, nTrial, parameters)
            else:
                nCount, confidence, confidenceRT, actual_duration, user_aborted, bpm, num_beats, modality = trial(condition, duration, nTrial, parameters, bpm)

            if user_aborted:
                break
            core.wait(0.5)
            parameters["triggers"]["trialStop"]()  # Send trigger or None
           # Store results in a DataFrame
            if parameters["results_df"] is None:
                parameters["results_df"] = pd.DataFrame(
                    {
                        "modality": [modality],
                        "nTrial": [nTrial],
                        "Reported": [nCount],
                        "Condition": [condition],
                        "Duration": [duration],
                        "Actual duration": [actual_duration],
                        "Confidence": [confidence],
                        "ConfidenceRT": [confidenceRT],
                        "bpm": [bpm],
                        "num_beats": [num_beats]
                    }
                )
            else:
                parameters["results_df"] = pd.concat(
                    [
                        parameters["results_df"],
                        pd.DataFrame(
                            {
                                "modality": [modality],
                                "nTrial": [nTrial],
                                "Reported": [nCount],
                                "Condition": [condition],
                                "Duration": [duration],
                                "Actual duration": [actual_duration],
                                "Confidence": [confidence],
                                "ConfidenceRT": [confidenceRT],
                                "bpm": [bpm],
                                "num_beats": [num_beats]
                            }
                        ),
                    ],
                    ignore_index=True,
                )

        task_end_time = core.getTime()
        task_duration = (task_end_time-task_start_time)/60

        if idx == 0:
            message = visual.TextStim(
                parameters["win"],
                text=parameters["texts"]["taskEnd1"],
                pos=(0.0, 0.2),
                height=parameters["textSize"],
                languageStyle=parameters['languageStyle'],
                wrapWidth=15
            )
            message.draw()

            press = visual.TextStim(
                parameters["win"],
                height=parameters["textSize"],
                text=parameters["texts"]["continue_text"],
                pos=(0.0, -0.4),
                languageStyle=parameters['languageStyle'],
                wrapWidth=15
            )
            press.draw()
            parameters["win"].flip()
            event.waitKeys(keyList=parameters["startKey"])

        else:
            message = visual.TextStim(
                parameters["win"],
                text=parameters["texts"]["taskEnd2"],
                pos=(0.0, 0.2),
                height=parameters["textSize"],
                languageStyle=parameters['languageStyle'],
                wrapWidth=15
            )
            message.draw()

            press = visual.TextStim(
                parameters["win"],
                height=parameters["textSize"],
                text=parameters["texts"]["continue_text"],
                pos=(0.0, -0.4),
                languageStyle=parameters['languageStyle'],
                wrapWidth=15
            )
            press.draw()
            parameters["win"].flip()
            event.waitKeys(keyList=parameters["startKey"])

        if parameters["exteroception"] == False:
            parameters['triggers'][('HBC_End')]()
        else:
            parameters['triggers'][('CTCT_End')]()
        time_data = {"task duration": task_duration}
        sub_num = parameters["participant"]
        filename = f"taskDuration_{task}_{sub_num}.json"
        filepath = os.path.join(parameters["resultPath"], filename)

        core.wait(0.5)

        with open(filepath, "w") as f:
            json.dump(time_data, f, indent=2)

        # Save results
        parameters["results_df"].to_csv(
            parameters["resultPath"]
            + "/"
            + task
            + "_final.csv",
            index=False,
        )

        tasks = pd.concat([tasks, parameters["results_df"]], ignore_index=True)
        parameters["results_df"] = None

    # End of the task
    if not user_aborted:
        tasks.to_csv(
            parameters["resultPath"]
            + "/"
            + "HBC_CTCT"
            + "_final.csv",
            index=False,
        )
        end = visual.TextStim(
            parameters["win"],
            height=parameters["textSize"],
            pos=(0.0, 0.2),
            text=parameters["texts"]["task_completion"],
            languageStyle=parameters['languageStyle'],
            wrapWidth=15
        )
        end.draw()
        parameters["win"].flip()
    core.wait(2)
    parameters["win"].close()
    core.quit()

    return user_aborted


def check_if_user_aborted(parameters: dict):
    keys = event.getKeys()
    if "escape" in keys:
        print("User abort")
        parameters["win"].close()
        core.quit()
        return True
    return False



def confidenceRatingTask(
        parameters: dict,
) -> Tuple[Optional[float], Optional[float], bool, Optional[float], bool]:
    """
    Parameters
    ----------
    parameters : dict
        Parameters dictionary.

    """

    print("...starting confidence rating.")

    # Initialise default values
    confidence, confidenceRT = None, None
    if check_if_user_aborted(parameters):
        return (0, 0, 0, 0, True)


    parameters["win"].mouseVisible = False
    message = visual.TextStim(
        parameters["win"],
        height=parameters["textSize"],
        pos=(0, 0.2),
        text=parameters["texts"]["confidence"],
        languageStyle=parameters['languageStyle'],
        wrapWidth=50
    )
    slider = visual.Slider(
        win=parameters["win"],
        name="slider",
        pos=(0, -0.2),
        size=(0.7, 0.1),
        granularity=1,
        ticks=(0, 100),
        style="rating",
        color="LightGray",
        flip=False, startValue=random.randint(30, 70)
    )

    text_labels = [
        visual.TextStim(parameters["win"], text=label, pos=pos, languageStyle=parameters['languageStyle'],
                        wrapWidth=50, height=parameters["textSize"]) for label, pos in
        zip(parameters["texts"]["VASlabels"], [(-0.35, -0.3), (0.35, -0.3)])]

    slider.marker.size = (0.03, 0.03)
    start_time = core.getTime()

    # Initialize response parameters
    key_times = {'num_4': None, 'num_6': None}  # Track when keys are pressed
    key_board = keyboard.Keyboard()
    key_board.clearEvents()
    while True:
        if check_if_user_aborted(parameters):
            return (0, 0, 0, 0, True)
        current_time = core.getTime()
        keys = key_board.getKeys(keyList=['num_4', 'num_6', 'return'], waitRelease=False, clear=False)
        # Check for keyboard input
        if keys is not None and len(keys) > 0:
            latest_key = keys[-1]
            if latest_key.duration is not None:
                key_board.clearEvents()
                continue
            if latest_key.name in key_times:
                duration = current_time - latest_key.tDown
                movement = int(duration * 5) + 1  # Increase speed over time
                if latest_key.name == 'num_4':
                    slider.markerPos -= movement
                elif latest_key.name == 'num_6':
                    slider.markerPos += movement

                # Ensure marker position stays within bounds
                if slider.markerPos < 0:
                    slider.markerPos = 0
                elif slider.markerPos > 100:
                    slider.markerPos = 100

                # Check if response provided
            if ('return' == latest_key.name) and (current_time - start_time > parameters["minRatingTime"]):
                confidence, confidenceRT, ratingProvided = (
                    slider.markerPos,
                    current_time - start_time,
                    True,
                )
                print(
                    f"... Confidence level: {confidence}"
                    + f" with response time {round(confidenceRT, 2)} seconds"
                )
                # Change marker color after response provided
                slider.marker.color = "green"
                for label in text_labels:
                    label.draw()
                slider.draw()
                message.draw()
                parameters["win"].flip()
                core.wait(0.2)
                if check_if_user_aborted(parameters):
                    return (0, 0, 0, 0, True)
                break
        elif current_time - start_time > parameters["maxRatingTime"]:  # if too long
            ratingProvided = False

            # Text feedback if no rating provided
            message = visual.TextStim(
                parameters["win"],
                height=parameters["textSize"],
                text="Too late",
                color="red",
                pos=(0.0, -0.2),
                languageStyle=parameters['languageStyle'],
                wrapWidth=50
            )
            message.draw()
            parameters["win"].flip()
            core.wait(0.5)
            if check_if_user_aborted(parameters):
                return (0, 0, 0, 0, True)
            break

        for label in text_labels:
            label.draw()
        slider.draw()
        message.draw()
        parameters["win"].flip()
    key_board.clearEvents()
    ratingEndTrigger = time.time()
    parameters["win"].flip()


    return confidence, confidenceRT, ratingProvided, ratingEndTrigger, False


def trial(
        condition: str,
        duration: int,
        nTrial: int,
        parameters: dict,
        bpm = None,
        num_beats = None
,
) -> Tuple[Optional[int], Optional[float], Optional[float], Optional[float], Optional[bool]]:
    """Run one trial.

    Parameters
    ----------
    condition : str
        The trial condition, can be `"Rest"` or `"Count"`.
    duration : int
        The lenght of the recording (in seconds).
    nTrial : int
        Trial number.
    parameters : dict
        Task parameters.

    Returns
    -------
    nCount : int
        The number of heartbeat estimated by the participant.
    confidence : int
        The confidence in the estimation of the heartbeat provided by the
        participant.
    confidenceRT : float
        The response time to provide confidence rating.

    """

    # Initialize default values
    confidence, confidenceRT = None, None
    nCounts: str = ""

    # Ask the participant to press 'Space' (default) to start the trial
    # messageStart = visual.TextStim(
    #     parameters["win"],
    #     height=parameters["textSize"],
    #     text=parameters["texts"]["continue_text"],
    #     languageStyle=parameters['languageStyle'],
    #     wrapWidth=15
    # )
    # messageStart.draw()
    # parameters["win"].flip()
    # event.waitKeys(keyList=parameters["startKey"])
    # parameters["win"].flip()

    is_oxi = parameters['data_stream_device'] == 'oxi'
    is_EEG = parameters['data_stream_device'] == 'EEG'
    if is_oxi:
        parameters["oxiTask"].setup()
        parameters["oxiTask"].read(duration=2)

  #  if (not parameters["exteroception"]) or (condition == "Training"):
    if  parameters["exteroception"] == False:
        modality = "Intero"
        # Show instructions
        if condition == "Rest":
            message = visual.TextStim(
                parameters["win"],
                text=parameters["texts"]["Rest"],
                pos=(0.0, 0.2),
                height=parameters["textSize"],
                languageStyle=parameters['languageStyle'],
                wrapWidth=15
            )
            message.draw()
            parameters["restLogo"].draw()

        elif condition == "Count":
            message = visual.TextStim(
                parameters["win"],
                text=parameters["texts"]["Count"],
                pos=(0.0, 0.2),
                height=parameters["textSize"],
                languageStyle=parameters['languageStyle'],
                wrapWidth=15
            )
            message.draw()
            parameters["heartLogoTrial"].draw()

        elif condition == "Training":
            message = visual.TextStim(
                parameters["win"],
                text=parameters["texts"]["Training"],
                pos=(0.0, 0.2),
                height=parameters["textSize"],
                languageStyle=parameters['languageStyle'],
                wrapWidth=15
            )
            message.draw()
            parameters["heartLogoTrain"].draw()
        parameters["win"].flip()

        # Wait for a beat to start the task
        if is_oxi:
            parameters["oxiTask"].waitBeat()


        # Sound signaling trial start
        if (condition == "Count") | (condition == "Training"):
            if is_oxi:
                parameters["oxiTask"].readInWaiting()
                # Add event marker
                parameters["oxiTask"].channels["Channel_0"][-1] = 1

            winsound.PlaySound(parameters["noteStart"],
                               winsound.SND_FILENAME)

            time_start = time.time()
            parameters["triggers"]["listeningStart"]()
        # core.wait(0.05)

        if is_oxi:
            # Record for a desired time length
            parameters["oxiTask"].read(duration=duration - 1)
        if is_EEG:
            core.wait(duration - 1)

        # Sound signaling trial stop
        if (condition == "Count") | (condition == "Training"):
            # Add event marker
            if is_oxi:
                parameters["oxiTask"].readInWaiting()
                parameters["oxiTask"].channels["Channel_0"][-1] = 2

            actual_duration = time.time() - time_start
            parameters["triggers"]["listeningStop"]()

            winsound.PlaySound(parameters["noteStop"], winsound.SND_FILENAME)

            if is_oxi:
                parameters["oxiTask"].readInWaiting()

        # Hide instructions
        parameters["win"].flip()
        core.wait(0.5)

        # Save recording
        if is_oxi:
            parameters["oxiTask"].save(
                parameters["resultPath"]
                + "/"
                + parameters["participant"]
                + str(nTrial)
                + "_"
                + str(nTrial)
            )
        ###############################
        # Record participant estimation
        ###############################
        if (condition == "Count") | (condition == "Training"):
            # Ask the participant to press '0' (default) to start the trial
            messageCount = visual.TextStim(
                parameters["win"],
                height=parameters["textSize"],
                pos=(0, 0.2),
                text=parameters["texts"]["nCount"],
                languageStyle=parameters['languageStyle'],
                wrapWidth=10
            )
            messageCount.draw()
            parameters["win"].flip()

            parameters["triggers"]["decisionStart"]()  # Send trigger or None

            nCounts = ""

            while True:

                # Record new key
                key = event.waitKeys(
                    keyList=[
                        "escape",
                        "backspace",
                        "return",
                        "num_1",
                        "num_2",
                        "num_3",
                        "num_4",
                        "num_5",
                        "num_6",
                        "num_7",
                        "num_8",
                        "num_9",
                        "num_0",
                    ]
                )

                if key[0] == "escape":
                    print("User abort")
                    return -1, -1.0, -1.0, -1.0, True
                if key[0] == "backspace":
                    if nCounts:
                        nCounts = nCounts[:-1]
                elif key[0] == "return":
                    if not all(char.isdigit() for char in nCounts):
                        messageError = visual.TextStim(
                            parameters["win"],
                            height=parameters["textSize"],
                            pos=(0, 0.2),
                            text=parameters["texts"]["not_number_input"],
                            languageStyle=parameters['languageStyle'],
                            wrapWidth=15
                        )
                        messageError.draw()
                        parameters["win"].flip()
                        core.wait(0.1)
                    elif nCounts == "":
                        messageError = visual.TextStim(
                            parameters["win"],
                            height=parameters["textSize"],
                            pos=(0, 0.2),
                            text=parameters["texts"]["not_number_input"],
                            languageStyle=parameters['languageStyle'],
                            wrapWidth=15
                        )
                        messageError.draw()
                        parameters["win"].flip()
                        core.wait(0.1)
                    else:
                        break

                else:
                    if key:
                        nCounts += [s for s in key[0] if s.isdigit()][0]

                # Show the text on the screen
                recordedText = visual.TextStim(
                    parameters["win"], height=parameters["textSize"], text=nCounts
                )
                recordedText.draw()
                messageCount.draw()
                parameters["win"].flip()

            parameters["triggers"]["decisionStop"]()  # Send trigger or None
            core.wait(0.5)
            ##############
            # Rating scale
            ##############
            if parameters["rating"] is True:
                (
                    confidence,
                    confidenceRT,
                    ratingProvided,
                    ratingEndTrigger, userAborted
                ) = confidenceRatingTask(parameters)


        finalCount = int(nCounts) if nCounts else -1

    if parameters["exteroception"] == True:

        modality = "Extero"

        if condition == "Count":
            message = visual.TextStim(
                parameters["win"],
                text=parameters["texts"]["CountExtero"],
                pos=(0.0, 0.2),
                height=parameters["textSize"],
                languageStyle=parameters['languageStyle'],
                wrapWidth=15
            )
            message.draw()
            parameters["listenLogoTrial"].draw()

        elif condition == "TrainingExtero":
            message = visual.TextStim(
                parameters["win"],
                text=parameters["texts"]["TrainingExtero"],
                pos=(0.0, 0.2),
                height=parameters["textSize"],
                languageStyle=parameters['languageStyle'],
                wrapWidth=15
            )
            message.draw()
            parameters["listenLogoTrial"].draw()

        parameters["win"].flip()

        # Wait for a beat to start the task
        if is_oxi:
            parameters["oxiTask"].waitBeat()


        # Sound signaling trial start
        if (condition == "Count") | (condition == "TrainingExtero"):
            if is_oxi:
                parameters["oxiTask"].readInWaiting()
                # Add event marker
                parameters["oxiTask"].channels["Channel_0"][-1] = 1

            winsound.PlaySound(parameters["noteStart"], winsound.SND_FILENAME)

            time_start = time.time()
            parameters["triggers"]["listeningStart"]()

            num_beats = repeat_tone_for_duration("cardioception/HBC/Sounds/heart_sounds/one_heartbeat.wav", total_duration_sec=duration, bpm=bpm)

        # Sound signaling trial stop
        if (condition == "Count") | (condition == "TrainingExtero"):
            # Add event marker
            if is_oxi:
                parameters["oxiTask"].readInWaiting()
                parameters["oxiTask"].channels["Channel_0"][-1] = 2

            actual_duration = time.time() - time_start
            parameters["triggers"]["listeningStop"]()

            winsound.PlaySound(parameters["noteStop"], winsound.SND_FILENAME)
            if is_oxi:
                parameters["oxiTask"].readInWaiting()
        # Hide instructions
        parameters["win"].flip()
        core.wait(0.5)
        # Save recording
        if is_oxi:
            parameters["oxiTask"].save(
                parameters["resultPath"]
                + "/"
                + parameters["participant"]
                + str(nTrial)
                + "_"
                + str(nTrial)
            )
        ###############################
        # Record participant estimation
        ###############################
        if (condition == "Count") | (condition == "TrainingExtero"):
            messageCount = visual.TextStim(
                parameters["win"],
                height=parameters["textSize"],
                pos=(0, 0.2),
                text=parameters["texts"]["nCount"],
                languageStyle=parameters['languageStyle'],
                wrapWidth=10
            )
            messageCount.draw()
            parameters["win"].flip()

            parameters["triggers"]["decisionStart"]()  # Send trigger or None

            nCounts = ""
            while True:

                # Record new key
                key = event.waitKeys(
                    keyList=[
                        "escape",
                        "backspace",
                        "return",
                        "num_1",
                        "num_2",
                        "num_3",
                        "num_4",
                        "num_5",
                        "num_6",
                        "num_7",
                        "num_8",
                        "num_9",
                        "num_0",
                    ]
                )

                if key[0] == "escape":
                    print("User abort")
                    return -1, -1.0, -1.0, -1.0, True
                if key[0] == "backspace":
                    if nCounts:
                        nCounts = nCounts[:-1]
                elif key[0] == "return":
                    if not all(char.isdigit() for char in nCounts):
                        messageError = visual.TextStim(
                            parameters["win"],
                            height=parameters["textSize"],
                            pos=(0, 0.2),
                            text=parameters["texts"]["not_number_input"],
                            languageStyle=parameters['languageStyle'],
                            wrapWidth=15
                        )
                        messageError.draw()
                        parameters["win"].flip()
                        core.wait(0.1)
                    elif nCounts == "":
                        messageError = visual.TextStim(
                            parameters["win"],
                            height=parameters["textSize"],
                            pos=(0, 0.2),
                            text=parameters["texts"]["not_number_input"],
                            languageStyle=parameters['languageStyle'],
                            wrapWidth=15
                        )
                        messageError.draw()
                        parameters["win"].flip()
                        core.wait(0.1)
                    else:
                        break

                else:
                    if key:
                        nCounts += [s for s in key[0] if s.isdigit()][0]

                # Show the text on the screen
                recordedText = visual.TextStim(
                    parameters["win"], height=parameters["textSize"], text=nCounts
                )
                recordedText.draw()
                messageCount.draw()
                parameters["win"].flip()

            parameters["triggers"]["decisionStop"]()  # Send trigger or None
            core.wait(0.5)
            ##############
            # Rating scale
            ##############
            if parameters["rating"] is True:
                (
                    confidence,
                    confidenceRT,
                    ratingProvided,
                    ratingEndTrigger, userAborted
                ) = confidenceRatingTask(parameters)

        finalCount = int(nCounts) if nCounts else -1


    return finalCount, -1 if confidence is None else confidence, -1 if confidenceRT is None else confidenceRT, actual_duration, False, -1 if bpm is None else bpm, -1 if num_beats is None else num_beats, modality


def tutorial(parameters: dict):
    """Run tutorial for the Heartbeat Counting Task.

    Parameters
    ----------
    parameters : dict
        Task parameters.
    win : `psychopy.visual.window` or None
        The window in which to draw objects.
    """

    from psychopy import event, visual

    # Tutorial 1
    messageStart = visual.TextStim(
        parameters["win"],
        height=parameters["textSize"],
        text=parameters["texts"]["Tutorial1"],
        languageStyle=parameters['languageStyle'],
        wrapWidth=15

    )
    messageStart.draw()
    press = visual.TextStim(
        parameters["win"],
        height=parameters["textSize"],
        text=parameters["texts"]["continue_text"],
        pos=(0.0, -0.4),
        languageStyle=parameters['languageStyle'],
        wrapWidth=15
    )
    press.draw()
    parameters["win"].flip()
    event.waitKeys(keyList=parameters["startKey"])

    # Tutorial 2
    messageStart = visual.TextStim(
        parameters["win"],
        height=parameters["textSize"],
        pos=(0.0, 0.2),
        text=parameters["texts"]["Tutorial2"],
        languageStyle=parameters['languageStyle'],
        wrapWidth=15
    )
    messageStart.draw()
    parameters["heartLogoTrain"].draw()

    press = visual.TextStim(
        parameters["win"],
        height=parameters["textSize"],
        text=parameters["texts"]["continue_text"],
        pos=(0.0, -0.4),
        languageStyle=parameters['languageStyle'],
        wrapWidth=15
    )
    press.draw()
    parameters["win"].flip()
    event.waitKeys(keyList=parameters["startKey"])

    # Tutorial 3

    messageStart = visual.TextStim(
        parameters["win"],
        height=parameters["textSize"],
        pos=(0.0, 0.2),
        text=parameters["texts"]["Tutorial3"],
        languageStyle=parameters['languageStyle'],
        wrapWidth=15
    )
    messageStart.draw()

    press = visual.TextStim(
        parameters["win"],
        height=parameters["textSize"],
        text=parameters["texts"]["continue_text"],
        pos=(0.0, -0.4),
        languageStyle=parameters['languageStyle'],
        wrapWidth=15
    )
    press.draw()
    parameters["win"].flip()
    event.waitKeys(keyList=parameters["startKey"])

    # Tutorial 4
    messageStart = visual.TextStim(
        parameters["win"],
        height=parameters["textSize"],
        pos=(0.0, 0.2),
        text=parameters["texts"]["Tutorial4"],
        languageStyle=parameters['languageStyle'],
        wrapWidth=15
    )
    messageStart.draw()
    parameters["listenLogoTrain"].draw()

    press = visual.TextStim(
        parameters["win"],
        height=parameters["textSize"],
        text=parameters["texts"]["continue_text"],
        pos=(0.0, -0.4),
        languageStyle=parameters['languageStyle'],
        wrapWidth=15
    )
    press.draw()
    parameters["win"].flip()
    event.waitKeys(keyList=parameters["startKey"])

    # Tutorial 5
    messageStart = visual.TextStim(
        parameters["win"],
        height=parameters["textSize"],
        text=parameters["texts"]["Tutorial5"],
        languageStyle=parameters['languageStyle'],
        wrapWidth=10
    )
    messageStart.draw()
    press = visual.TextStim(
        parameters["win"],
        height=parameters["textSize"],
        text=parameters["texts"]["continue_text"],
        pos=(0.0, -0.4),
        languageStyle=parameters['languageStyle'],
        wrapWidth=15
    )
    press.draw()
    parameters["win"].flip()
    event.waitKeys(keyList=parameters["startKey"])

    # Tutorial 6
    messageStart = visual.TextStim(
        parameters["win"],
        height=parameters["textSize"],
        text=parameters["texts"]["Tutorial6"],
        languageStyle=parameters['languageStyle'],
        wrapWidth=15
    )
    messageStart.draw()
    press = visual.TextStim(
        parameters["win"],
        height=parameters["textSize"],
        text=parameters["texts"]["continue_text"],
        pos=(0.0, -0.4),
        languageStyle=parameters['languageStyle'],
        wrapWidth=15
    )
    press.draw()
    parameters["win"].flip()
    event.waitKeys(keyList=parameters["startKey"])


    # Tutorial 7
    messageStart = visual.TextStim(
        parameters["win"],
        height=parameters["textSize"],
        text=parameters["texts"]["Tutorial7"],
        languageStyle=parameters['languageStyle'],
        wrapWidth=15
    )
    messageStart.draw()
    press = visual.TextStim(
        parameters["win"],
        height=parameters["textSize"],
        text=parameters["texts"]["continue_text"],
        pos=(0.0, -0.4),
        languageStyle=parameters['languageStyle'],
        wrapWidth=15
    )
    press.draw()
    parameters["win"].flip()
    event.waitKeys(keyList=parameters["startKey"])


    # Tutorial 8
    messageStart = visual.TextStim(
        parameters["win"],
        height=parameters["textSize"],
        text=parameters["texts"]["Tutorial8"],
        languageStyle=parameters['languageStyle'],
        wrapWidth=15
    )
    messageStart.draw()
    press = visual.TextStim(
        parameters["win"],
        height=parameters["textSize"],
        text=parameters["texts"]["continue_text"],
        pos=(0.0, -0.4),
        languageStyle=parameters['languageStyle'],
        wrapWidth=15
    )
    press.draw()
    parameters["win"].flip()
    event.waitKeys(keyList=parameters["startKey"])

    # Practice trial
    messageStart = visual.TextStim(
        parameters["win"],
        height=parameters["textSize"],
        text=parameters["texts"]["HBC_Start"],
        languageStyle=parameters['languageStyle'],
        wrapWidth=15
    )
    messageStart.draw()
    press = visual.TextStim(
        parameters["win"],
        height=parameters["textSize"],
        text=parameters["texts"]["continue_text"],
        pos=(0.0, -0.4),
        languageStyle=parameters['languageStyle'],
        wrapWidth=15
    )
    press.draw()
    parameters["win"].flip()
    event.waitKeys(keyList=parameters["startKey"])

    parameters["exteroception"] = False
    _ = trial("Training", 15, 0, parameters)

    messageStart = visual.TextStim(
        parameters["win"],
        height=parameters["textSize"],
        text=parameters["texts"]["CTCT_Start"],
        languageStyle=parameters['languageStyle'],
        wrapWidth=15
    )
    messageStart.draw()
    press = visual.TextStim(
        parameters["win"],
        height=parameters["textSize"],
        text=parameters["texts"]["continue_text"],
        pos=(0.0, -0.4),
        languageStyle=parameters['languageStyle'],
        wrapWidth=15
    )
    press.draw()
    parameters["win"].flip()
    event.waitKeys(keyList=parameters["startKey"])

    parameters["exteroception"] = True
    _ = trial("TrainingExtero", 15, 0, parameters, bpm = 60)


    # Tutorial 9
    messageStart = visual.TextStim(
        parameters["win"],
        height=parameters["textSize"],
        text=parameters["texts"]["Tutorial9"],
        languageStyle=parameters['languageStyle'],
        wrapWidth=15
    )
    messageStart.draw()
    press = visual.TextStim(
        parameters["win"],
        height=parameters["textSize"],
        text=parameters["texts"]["continue_text"],
        pos=(0.0, -0.4),
        languageStyle=parameters['languageStyle'],
        wrapWidth=15
    )
    press.draw()
    parameters["win"].flip()
    event.waitKeys(keyList=parameters["startKey"])


   # Tutorial 10
    messageStart = visual.TextStim(
        parameters["win"],
        height=parameters["textSize"],
        text=parameters["texts"]["Tutorial10"],
        languageStyle=parameters['languageStyle'],
        wrapWidth=15
    )
    messageStart.draw()
    press = visual.TextStim(
        parameters["win"],
        height=parameters["textSize"],
        text=parameters["texts"]["continue_text"],
        pos=(0.0, -0.4),
        languageStyle=parameters['languageStyle'],
        wrapWidth=15
    )
    press.draw()
    parameters["win"].flip()
    event.waitKeys(keyList=parameters["startKey"])

# def tutorialExtero(parameters: dict):
#     """Run tutorial for the Heartbeat Counting Task.
#
#     Parameters
#     ----------
#     parameters : dict
#         Task parameters.
#     win : `psychopy.visual.window` or None
#         The window in which to draw objects.
#     """
#
#     # Tutorial 1
#     messageStart = visual.TextStim(
#         parameters["win"],
#         height=parameters["textSize"],
#         text=parameters["texts"]["TutorialExtero1"],
#         languageStyle=parameters['languageStyle'],
#         wrapWidth=15
#
#     )
#     messageStart.draw()
#     press = visual.TextStim(
#         parameters["win"],
#         height=parameters["textSize"],
#         text=parameters["texts"]["continue_text"],
#         pos=(0.0, -0.4),
#         languageStyle=parameters['languageStyle'],
#         wrapWidth=15
#     )
#     press.draw()
#     parameters["win"].flip()
#     event.waitKeys(keyList=parameters["startKey"])
#
#     # Tutorial 2
#     messageStart = visual.TextStim(
#         parameters["win"],
#         height=parameters["textSize"],
#         pos=(0.0, 0.2),
#         text=parameters["texts"]["TutorialExtero2"],
#         languageStyle=parameters['languageStyle'],
#         wrapWidth=15
#     )
#     messageStart.draw()
#     parameters["listenLogoTrain"].draw()
#
#     press = visual.TextStim(
#         parameters["win"],
#         height=parameters["textSize"],
#         text=parameters["texts"]["continue_text"],
#         pos=(0.0, -0.4),
#         languageStyle=parameters['languageStyle'],
#         wrapWidth=15
#     )
#     press.draw()
#     parameters["win"].flip()
#     event.waitKeys(keyList=parameters["startKey"])
#
#
#     # Tutorial 4
#     messageStart = visual.TextStim(
#         parameters["win"],
#         height=parameters["textSize"],
#         text=parameters["texts"]["TutorialExtero4"],
#         languageStyle=parameters['languageStyle'],
#         wrapWidth=10
#     )
#     messageStart.draw()
#     press = visual.TextStim(
#         parameters["win"],
#         height=parameters["textSize"],
#         text=parameters["texts"]["continue_text"],
#         pos=(0.0, -0.4),
#         languageStyle=parameters['languageStyle'],
#         wrapWidth=15
#     )
#     press.draw()
#     parameters["win"].flip()
#
#     event.waitKeys(keyList=parameters["startKey"])
#
#     # Tutorial 6
#     messageStart = visual.TextStim(
#         parameters["win"],
#         height=parameters["textSize"],
#         text=parameters["texts"]["TutorialExtero6"],
#         languageStyle=parameters['languageStyle'],
#         wrapWidth=15
#     )
#     messageStart.draw()
#     press = visual.TextStim(
#         parameters["win"],
#         height=parameters["textSize"],
#         text=parameters["texts"]["continue_text"],
#         pos=(0.0, -0.4),
#         languageStyle=parameters['languageStyle'],
#         wrapWidth=15
#     )
#     press.draw()
#     parameters["win"].flip()
#     event.waitKeys(keyList=parameters["startKey"])
#
#
#     # Tutorial 7
#     messageStart = visual.TextStim(
#         parameters["win"],
#         height=parameters["textSize"],
#         text=parameters["texts"]["TutorialExtero7"],
#         languageStyle=parameters['languageStyle'],
#         wrapWidth=15
#     )
#     messageStart.draw()
#     press = visual.TextStim(
#         parameters["win"],
#         height=parameters["textSize"],
#         text=parameters["texts"]["continue_text"],
#         pos=(0.0, -0.4),
#         languageStyle=parameters['languageStyle'],
#         wrapWidth=15
#     )
#     press.draw()
#     parameters["win"].flip()
#     event.waitKeys(keyList=parameters["startKey"])
#
#     # Practice trial
#     _ = trial("TrainingExtero", 15, 0, parameters, bpm=60)
#
#     # Tutorial 8
#     messageStart = visual.TextStim(
#         parameters["win"],
#         height=parameters["textSize"],
#         text=parameters["texts"]["TutorialExtero8"],
#         languageStyle=parameters['languageStyle'],
#         wrapWidth=15
#     )
#     messageStart.draw()
#     press = visual.TextStim(
#         parameters["win"],
#         height=parameters["textSize"],
#         text=parameters["texts"]["continue_text"],
#         pos=(0.0, -0.4),
#         languageStyle=parameters['languageStyle'],
#         wrapWidth=15
#     )
#     press.draw()
#     parameters["win"].flip()
#     event.waitKeys(keyList=parameters["startKey"])

def rest(parameters: dict, duration: float = 300.0):
    """Run a resting state period for heart rate variability before running the Heart
    Beat Counting Task.

    Parameters
    ----------
    parameters : dict
        Task parameters.
    duration : float
        Duration or the recording (seconds).

    """

    is_oxi = parameters['data_stream_device'] == 'oxi'
    is_EEG = parameters['data_stream_device'] == 'EEG'
    # Show the resting state instructions
    messageStart = visual.TextStim(
        parameters["win"],
        height=parameters["textSize"],
        pos=(0.0, 0.2),
        text=("Calibrating... Please sit quietly" " until the end of the recording."),
    )
    messageStart.draw()
    parameters["restLogo"].draw()
    parameters["win"].flip()

    # Record PPG signal
    if is_oxi:
        parameters["oxiTask"].setup()
        parameters["oxiTask"].read(duration=duration)

        # Save recording
        parameters["oxiTask"].save(
            parameters["resultPath"] + "/" + parameters["participant"] + "_Rest"
        )

    parameters['triggers']['restStart']()
    core.wait(duration)
    parameters['triggers']['restEnd']()

# delete
# def play_random_sound_looped(sound_path: str, duration: float) -> int:
#
#     """
#     Play a .wav file in a loop for a set duration.
#
#     Args:
#         sound_path (str): Full path to a .wav file (e.g., "123bpm.wav")
#         duration (float): Trial duration in seconds
#
#     Returns:
#         int: BPM value extracted from the filename
#     """
#
#     filename = os.path.basename(sound_path)
#     # Extract BPM from filename (e.g., "80bpm.wav")
#     match = re.search(r"(\d+)bpm", filename.lower())
#     bpm = int(match.group(1)) if match else None
#
#
#     trial_start_time = time.time()
#     while True:
#         elapsed = time.time() - trial_start_time
#         remaining = duration - elapsed
#         if remaining <= 0:
#             break
#
#         audio = AudioSegment.from_wav(sound_path)
#         sound_duration_sec = len(audio) / 1000
#
#         print(f"Played: {filename} | BPM: {bpm} | in duration of : {round(duration, 2)}  seconds")
#
#         play_obj = sa.play_buffer(
#             audio.raw_data,
#             num_channels=audio.channels,
#             bytes_per_sample=audio.sample_width,
#             sample_rate=audio.frame_rate,
#         )
#
#         if remaining >= sound_duration_sec:
#             time.sleep(sound_duration_sec)
#         else:
#
#             time.sleep(remaining)
#             play_obj.stop()
#             break
#
# #        audio.stop()
#
#     print("\n Sound stoped.")
#
#     return bpm

def ensure_numlock_on():
    # הפונקציה מדליקה את NumLock אם הוא כבוי (רק ב-Windows)
    VK_NUMLOCK = 0x90
    SCANCODE = 0x45  # Scan code for NumLock key

    # בדיקה אם NumLock כבוי
    if ctypes.windll.user32.GetKeyState(VK_NUMLOCK) == 0:
        # Key down
        ctypes.windll.user32.keybd_event(VK_NUMLOCK, SCANCODE, 0x1, 0)
        # Key up
        ctypes.windll.user32.keybd_event(VK_NUMLOCK, SCANCODE, 0x3, 0)