# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
import random
import time
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from psychopy import visual, core, event
from psychopy.hardware import keyboard


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


    # Run tutorial
    if runTutorial is True:
        tutorial(parameters)

    # Rest
    if parameters["restPeriod"] is True:
        rest(parameters, duration=parameters["restLength"])

    user_aborted = False

    for condition, duration, nTrial in zip(
            parameters["conditions"],
            parameters["times"],
            range(0, len(parameters["conditions"])),
    ):


        parameters['triggers']['trialStart']()

        nCount, confidence, confidenceRT, actual_duration, user_aborted = trial(condition, duration, nTrial, parameters)

        if user_aborted:
            break

        parameters["triggers"]["trialStop"]()  # Send trigger or None

        # Store results in a DataFrame
        if parameters["results_df"] is None:
            parameters["results_df"] = pd.DataFrame(
                {
                    "nTrial": [nTrial],
                    "Reported": [nCount],
                    "Condition": [condition],
                    "Duration": [duration],
                    "Actual duration": [actual_duration],
                    "Confidence": [confidence],
                    "ConfidenceRT": [confidenceRT],
                }
            )
        else:
            parameters["results_df"] = pd.concat(
                [
                    parameters["results_df"],
                    pd.DataFrame(
                        {
                            "nTrial": [nTrial],
                            "Reported": [nCount],
                            "Condition": [condition],
                            "Duration": [duration],
                            "Actual duration": [actual_duration],
                            "Confidence": [confidence],
                            "ConfidenceRT": [confidenceRT],
                        }
                    ),
                ],
                ignore_index=True,
            )

        # Save the results at each iteration
        parameters["results_df"].to_csv(
            parameters["resultPath"]
            + "/"
            + parameters["participant"]
            + parameters["session"]
            + ".csv",
            index=False,
        )

    # Save results
    parameters["results_df"].to_csv(
        parameters["resultPath"]
        + "/"
        + parameters["participant"]
        + parameters["session"]
        + "_final.csv",
        index=False,
    )
    # End of the task
    if not user_aborted:
        end = visual.TextStim(
            parameters["win"],
            height=parameters["textSize"],
            pos=(0.0, 0.0),
            text=parameters["texts"]["task_completion"],
        )
        end.draw()
        parameters["win"].flip()
    core.wait(3)
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
) -> Tuple[float, float, bool, float, bool]:
    """Confidence rating scale, using keyboard or mouse inputs.

    Parameters
    ----------
    parameters : dict
        Parameters dictionary.

    """

    print("...starting confidence rating.")

    # Initialise default values
    confidence, confidenceRT = None, None
    if check_if_user_aborted(parameters):
        return -1.0, -1.0, False, -1.0, True
    parameters["triggers"]["confidenceStart"]()
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
    key_times = {'left': None, 'right': None}  # Track when keys are pressed
    key_board = keyboard.Keyboard()
    key_board.clearEvents()
    while True:
        if check_if_user_aborted(parameters):
            return -1.0, -1.0, False, -1.0, True
        current_time = core.getTime()
        keys = key_board.getKeys(keyList=['right', 'left', 'space'], waitRelease=False, clear=False)
        # Check for keyboard input
        if keys is not None and len(keys) > 0:
            latest_key = keys[-1]
            if latest_key.duration is not None:
                key_board.clearEvents()
                continue
            if latest_key.name in key_times:
                duration = current_time - latest_key.tDown
                movement = int(duration * 5) + 1  # Increase speed over time
                if latest_key.name == 'left':
                    slider.markerPos -= movement
                elif latest_key.name == 'right':
                    slider.markerPos += movement

                # Ensure marker position stays within bounds
                if slider.markerPos < 0:
                    slider.markerPos = 0
                elif slider.markerPos > 100:
                    slider.markerPos = 100

                # Check if response provided
            if ('space' == latest_key.name) and (current_time - start_time > parameters["minRatingTime"]):
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
                    return -1.0, -1.0, False, -1.0, True
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
                return -1.0, -1.0, False, -1.0, True
            break

        for label in text_labels:
            label.draw()
        slider.draw()
        message.draw()
        parameters["win"].flip()
    key_board.clearEvents()
    ratingEndTrigger = time.time()
    parameters["win"].flip()
    parameters["triggers"]["confidenceStop"]()
    return confidence, confidenceRT, ratingProvided, ratingEndTrigger, False


def trial(
        condition: str,
        duration: int,
        nTrial: int,
        parameters: dict,
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

    from psychopy import core, event, visual
    # Initialize default values
    confidence, confidenceRT = None, None
    nCounts: str = ""

    # Ask the participant to press 'Space' (default) to start the trial
    messageStart = visual.TextStim(
        parameters["win"], height=parameters["textSize"], text=parameters["texts"]["continue_text"]
    )
    messageStart.draw()
    parameters["win"].flip()
    event.waitKeys(keyList=parameters["startKey"])
    parameters["win"].flip()
    is_oxi = parameters['data_stream_device'] == 'oxi'
    is_EEG = parameters['data_stream_device'] == 'EEG'
    if is_oxi:
        parameters["oxiTask"].setup()
        parameters["oxiTask"].read(duration=2)

    # Show instructions
    if condition == "Rest":
        message = visual.TextStim(
            parameters["win"],
            text=parameters["texts"]["Rest"],
            pos=(0.0, 0.2),
            height=parameters["textSize"],
        )
        message.draw()
        parameters["restLogo"].draw()
    elif (condition == "Count") | (condition == "Training"):
        message = visual.TextStim(
            parameters["win"],
            text=parameters["texts"]["Count"],
            pos=(0.0, 0.2),
            height=parameters["textSize"],
        )
        message.draw()
        parameters["heartLogo"].draw()
    parameters["win"].flip()

    # Wait for a beat to start the task
    if is_oxi:
        parameters["oxiTask"].waitBeat()
    core.wait(3)

    # Sound signaling trial start
    if (condition == "Count") | (condition == "Training"):
        if is_oxi:
            parameters["oxiTask"].readInWaiting()
            # Add event marker
            parameters["oxiTask"].channels["Channel_0"][-1] = 1
        parameters["noteStart"].play()

    parameters["triggers"]["listeningStart"]()
    core.wait(1)
    time_start = time.time()
    if is_oxi:
        # Record for a desired time length
        parameters["oxiTask"].read(duration=duration - 1)
    if is_EEG:
        core.wait(duration - 1)
    actual_duration = time.time() - time_start
    # Sound signaling trial stop
    if (condition == "Count") | (condition == "Training"):
        # Add event marker
        if is_oxi:
            parameters["oxiTask"].readInWaiting()
            parameters["oxiTask"].channels["Channel_0"][-1] = 2
        parameters["noteStop"].play()

        if is_oxi:
            parameters["oxiTask"].readInWaiting()

    parameters["triggers"]["listeningStop"]()
    core.wait(3)
    # Hide instructions
    parameters["win"].flip()

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
        # Ask the participant to press 'Space' (default) to start the trial
        messageCount = visual.TextStim(
            parameters["win"],
            height=parameters["textSize"],
            pos=(0, 0.2),
            text=parameters["texts"]["nCount"],
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
                    "1",
                    "2",
                    "3",
                    "4",
                    "5",
                    "6",
                    "7",
                    "8",
                    "9",
                    "0",
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
                    )
                    messageError.draw()
                    parameters["win"].flip()
                    core.wait(2)
                elif nCounts == "":
                    messageError = visual.TextStim(
                        parameters["win"],
                        height=parameters["textSize"],
                        pos=(0, 0.2),
                        text=parameters["texts"]["not_number_input"],
                    )
                    messageError.draw()
                    parameters["win"].flip()
                    core.wait(2)
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

    return finalCount, -1 if confidence is None else confidence, -1 if confidenceRT is None else confidenceRT, actual_duration, False


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
    )
    messageStart.draw()
    press = visual.TextStim(
        parameters["win"],
        height=parameters["textSize"],
        text=parameters["texts"]["continue_text"],
        pos=(0.0, -0.4),
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
    )
    messageStart.draw()
    parameters["heartLogo"].draw()
    press = visual.TextStim(
        parameters["win"],
        height=parameters["textSize"],
        text=parameters["texts"]["continue_text"],
        pos=(0.0, -0.4),
    )
    press.draw()
    parameters["win"].flip()
    event.waitKeys(keyList=parameters["startKey"])

    # Tutorial 3
    if parameters["taskVersion"] == "Shandry":
        messageStart = visual.TextStim(
            parameters["win"],
            height=parameters["textSize"],
            pos=(0.0, 0.2),
            text=parameters["texts"]["Tutorial3"],
        )
        messageStart.draw()
        parameters["restLogo"].draw()
        press = visual.TextStim(
            parameters["win"],
            height=parameters["textSize"],
            text=parameters["texts"]["continue_text"],
            pos=(0.0, -0.4),
        )
        press.draw()
        parameters["win"].flip()
        event.waitKeys(keyList=parameters["startKey"])

    # Tutorial 4
    messageStart = visual.TextStim(
        parameters["win"],
        height=parameters["textSize"],
        text=parameters["texts"]["Tutorial4"],
    )
    messageStart.draw()
    press = visual.TextStim(
        parameters["win"],
        height=parameters["textSize"],
        text=parameters["texts"]["continue_text"],
        pos=(0.0, -0.4),
    )
    press.draw()
    parameters["win"].flip()

    event.waitKeys(keyList=parameters["startKey"])

    # Tutorial 5
    messageStart = visual.TextStim(
        parameters["win"],
        height=parameters["textSize"],
        text=parameters["texts"]["Tutorial5"],
    )
    messageStart.draw()
    press = visual.TextStim(
        parameters["win"],
        height=parameters["textSize"],
        text=parameters["texts"]["continue_text"],
        pos=(0.0, -0.4),
    )
    press.draw()
    parameters["win"].flip()
    event.waitKeys(keyList=parameters["startKey"])

    # Tutorial 6
    messageStart = visual.TextStim(
        parameters["win"],
        height=parameters["textSize"],
        text=parameters["texts"]["Tutorial6"],
    )
    messageStart.draw()
    press = visual.TextStim(
        parameters["win"],
        height=parameters["textSize"],
        text=parameters["texts"]["continue_text"],
        pos=(0.0, -0.4),
    )
    press.draw()
    parameters["win"].flip()
    event.waitKeys(keyList=parameters["startKey"])

    # Tutorial 7
    messageStart = visual.TextStim(
        parameters["win"],
        height=parameters["textSize"],
        text=parameters["texts"]["Tutorial7"],
    )
    messageStart.draw()
    press = visual.TextStim(
        parameters["win"],
        height=parameters["textSize"],
        text=parameters["texts"]["continue_text"],
        pos=(0.0, -0.4),
    )
    press.draw()
    parameters["win"].flip()
    event.waitKeys(keyList=parameters["startKey"])

    # Tutorial 8
    messageStart = visual.TextStim(
        parameters["win"],
        height=parameters["textSize"],
        text=parameters["texts"]["Tutorial8"],
    )
    messageStart.draw()
    press = visual.TextStim(
        parameters["win"],
        height=parameters["textSize"],
        text=parameters["texts"]["continue_text"],
        pos=(0.0, -0.4),
    )
    press.draw()
    parameters["win"].flip()
    event.waitKeys(keyList=parameters["startKey"])

    # Practice trial
    _ = trial("Count", 15, 0, parameters)

    # Tutorial 9
    messageStart = visual.TextStim(
        parameters["win"],
        height=parameters["textSize"],
        text=parameters["texts"]["Tutorial9"],
    )
    messageStart.draw()
    press = visual.TextStim(
        parameters["win"],
        height=parameters["textSize"],
        text=parameters["texts"]["continue_text"],
        pos=(0.0, -0.4),
    )
    press.draw()
    parameters["win"].flip()
    event.waitKeys(keyList=parameters["startKey"])


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
