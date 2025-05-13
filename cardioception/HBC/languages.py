# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
from typing import Collection, Dict


def english(exteroception: bool) -> Dict[str, Collection[str]]:
    """Create the text dictionary with instruction in English

    Parameters
    ----------
    exteroception : bool
        If `True`, the task includes and exteroceptive control condition.

    Returns
    -------
    texts : dict

    """
    texts = {
        "Rest": "Please sit quietly until the next session",
        "Count": (
            "After you hear START, try to count your heartbeats"
            " by concentrating on your body feelings."
            " Stop counting when you hear STOP"
        ),
        "VASlabels": ["Guess", "Certain"],
        "Training": (
            "After you hear START, try to count your heartbeats"
            " by concentrating on your body feelings"
            " Stop counting when you hear STOP"
        ),
        "nCount": (
            "How many heartbeats did you count?"
            " Write a number and press ENTER to validate."
        ),
        "confidence": (
            "How confident are you about your count?"
            "Use the RIGHT/LEFT keys to select and the SPACE key to confirm"
        ),
    }

    texts[
        "Tutorial1"
    ] = (
        "During this experiment, we will ask you to silently"
        " count your heartbeats for different intervals of time."
    )

    texts[
        "Tutorial2"
    ] = (
        'When you see this "heart" icon, you will silently count your'
        " heartbeats by focusing on your body sensations."
    )

    texts[
        "Tutorial3"
    ] = (
        'Sometime, you will also encounter this "rest" icon.'
        " In this case your task will just be to sit quietly until the next"
        " session."
    )
    texts['continue_text'] = "Please press SPACE to continue"
    texts['not_number_input'] = "You should only provide numbers"
    texts['task_completion'] = "You have completed the task. Thank you for your participation."
    if exteroception is True:
        moreResp = "UP key"
        lessResp = "DOWN key"
        texts[
            "Tutorial3bis"
        ] = """For some trials, instead of seeing the heart icon, you will see a listening icon. You will then have to listen to a first set of beeps, instead of your heart."""

        texts[
            "Tutorial3ter"
        ] = f"""After these first beeps, you will see the response icons appear, and a second set of beeps will play.

                As quickly and accurately as possible, you will listen to these beeps and decide if they are faster ({moreResp}) or slower ({lessResp}) than the first set of beeps.
                
                The second series of beeps will ALWAYS be slower or faster than the first series. Please guess, even if you are unsure."""

    texts["Tutorial4"] = (
        "The beginning and the end of the task will be signalled when you hear"
        " the words 'START'' and 'STOP'. While counting your heartbeats, you"
        " may close your eyes if you find that helpful. Please keep your hand"
        " still during the counting period, to avoid interfering with"
        " the heartbeat recording."
    )
    texts["Tutorial5"] = (
        "After the counting part of the task, you will be asked to report the"
        " exact number of heartbeats you felt during the interval between"
        " 'START' and 'STOP'. Please do not try to estimate the number of"
        " heartbeats, but instead only report the heartbeats you actually felt"
        " during the interval. You will input your response using the number"
        " pad and press return when done. You can also correct your response"
        " using backspace."
    )

    texts["Tutorial6"] = (
        "Once you have made your response, you will estimate your subjective"
        " feeling of confidence in how accurate your count was"
        " for that interval. A large number here means that you are totally"
        " certain you counted the exact number of heartbeats that occured,"
        " and a small number means that you are totally uncertain or felt that"
        " you were guessing about the"
        " number of heartbeats. You should use the RIGHT and LEFT"
        " key to select your response and the DOWN key to confirm."
    )
    texts["Tutorial7"] = (
        "Before the main task begins there is a short resting period of"
        " several minutes, during which we will calibrate the heartbeat"
        " recording. During this period, please sit quietly with your"
        " hands still to avoid interfering with the calibration."
        " Afterwards, the counting task will begin, and will take about"
        " 6 minutes in total."
    )
    texts["Tutorial8"] = (
        "You will now complete a short practice task."
        " Please ask the experimenter if you have any questions before"
        " continuing to the main experiment."
    )
    texts["Tutorial9"] = (
        "Good job! If you have any question, ask the experimenter now,"
        " otherwise press SPACE to continue to the experiment."
    )
    return texts


def hebrew(exteroception: bool) -> Dict[str, Collection[str]]:
    """Create the text dictionary with instruction in Hebrew

    Parameters
    ----------
    device : str
        Can be `"keyboard"` or `"mouse"`.
    setup : str
        The experimental setup. Can be `"behavioral"` or `"test"`.
    exteroception : bool
        If `True`, the task includes and exteroceptive control condition.

    Returns
    -------
    texts : dict

    """

    texts = {
        "Rest": "Please sit quietly until the next session",
        "Count": (
            "After you hear START, try to count your heartbeats"
            " by concentrating on your body feelings."
            " Stop counting when you hear STOP"
        ),
        "Training": (
            "After you hear START, try to count your heartbeats"
            " by concentrating on your body feelings"
            " Stop counting when you hear STOP"
        ),
        "nCount": (
            "How many heartbeats did you count?"
            " Write a number and press ENTER to validate."
        ),
        "VASlabels": ["Guess", "Certain"],
        "confidence": (
            "How confident are you about your count?"
            "Use the RIGHT/LEFT keys to select and the DOWN key to confirm"
        ),
    }

    texts[
        "Tutorial1"
    ] = (
        "During this experiment, we will ask you to silently"
        " count your heartbeats for different intervals of time."
    )

    texts[
        "Tutorial2"
    ] = (
        'When you see this "heart" icon, you will silently count your'
        " heartbeats by focusing on your body sensations."
    )

    texts[
        "Tutorial3"
    ] = (
        'Sometime, you will also encounter this "rest" icon.'
        " In this case your task will just be to sit quietly until the next"
        " session."
    )
    texts['continue_text'] = "Please press SPACE to continue"
    texts['not_number_input'] = "You should only provide numbers"
    texts['task_completion'] = "You have completed the task. Thank you for your participation."
    if exteroception is True:
        moreResp = "UP key"
        lessResp = "DOWN key"
        texts[
            "Tutorial3bis"
        ] = """For some trials, instead of seeing the heart icon, you will see a listening icon. You will then have to listen to a first set of beeps, instead of your heart."""

        texts[
            "Tutorial3ter"
        ] = f"""After these first beeps, you will see the response icons appear, and a second set of beeps will play.

                    As quickly and accurately as possible, you will listen to these beeps and decide if they are faster ({moreResp}) or slower ({lessResp}) than the first set of beeps.

                    The second series of beeps will ALWAYS be slower or faster than the first series. Please guess, even if you are unsure."""

    texts["Tutorial4"] = (
        "The beginning and the end of the task will be signalled when you hear"
        " the words 'START'' and 'STOP'. While counting your heartbeats, you"
        " may close your eyes if you find that helpful. Please keep your hand"
        " still during the counting period, to avoid interfering with"
        " the heartbeat recording."
    )
    texts["Tutorial5"] = (
        "After the counting part of the task, you will be asked to report the"
        " exact number of heartbeats you felt during the interval between"
        " 'START' and 'STOP'. Please do not try to estimate the number of"
        " heartbeats, but instead only report the heartbeats you actually felt"
        " during the interval. You will input your response using the number"
        " pad and press return when done. You can also correct your response"
        " using backspace."
    )

    texts["Tutorial6"] = (
        "Once you have made your response, you will estimate your subjective"
        " feeling of confidence in how accurate your count was"
        " for that interval. A large number here means that you are totally"
        " certain you counted the exact number of heartbeats that occured,"
        " and a small number means that you are totally uncertain or felt that"
        " you were guessing about the"
        " number of heartbeats. You should use the RIGHT and LEFT"
        " key to select your response and the DOWN key to confirm."
    )
    texts["Tutorial7"] = (
        "Before the main task begins there is a short resting period of"
        " several minutes, during which we will calibrate the heartbeat"
        " recording. During this period, please sit quietly with your"
        " hands still to avoid interfering with the calibration."
        " Afterwards, the counting task will begin, and will take about"
        " 6 minutes in total."
    )
    texts["Tutorial8"] = (
        "You will now complete a short practice task."
        " Please ask the experimenter if you have any questions before"
        " continuing to the main experiment."
    )
    texts["Tutorial9"] = (
        "Good job! If you have any question, ask the experimenter now,"
        " otherwise press SPACE to continue to the experiment."
    )
    return texts
