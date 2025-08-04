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


def hebrew(CTCT: bool) -> Dict[str, Collection[str]]:
    """Create the text dictionary with instruction in Hebrew

    Parameters
    ----------
    device : str
        Can be `"keyboard"` or `"mouse"`.
    setup : str
        The experimental setup. Can be `"behavioral"` or `"test"`.
    ctct : bool
        If `True`, the task includes and exteroceptive control condition.

    Returns
    -------
    texts : dict

    """

    texts = {

        "Rest": "נא לשבת בשקט עד לתחילת החלק הבא.",

        "Count": (
        """      
        """
        ),

        "CountExtero":(
            """
            """
        ),
        "VASlabels": ["ניחוש", "בטוח"],

        "Training": (
            """
          אחרי שתשמע/י "Start", נסה/י לספור את פעימות הלב שלך. 
          יש להסתמך על תחושות הגוף. 
          הפסיק/י לספור כאשר יושמע "Stop"
            """
        ),

        "TrainingExtero": (
            """
          אחרי שתשמע/י "Start", נסה/י לספור את הסאונד המדמה פעימות של הלב. 
          הפסיק/י לספור כאשר יושמע "Stop"
            """
        ),

        "confidence": (
            """
            עד כמה את/ה בטוח/ה בהחלטה שלך?
             ללחוץ Enter להמשך.
        
        """
        ),

        "nCount": (
            """
            כמה פעימות לב ספרת?
            לרשום את המספר ולהקיש Enter להמשך.
            """

        ),
        "TrainingConfidence": (
            """
            עד כמה את/ה בטוח/ה בהחלטה שלך? 
            ללחוץ 6 (ימינה) או 4 (שמאלה) כדי להזיז הסמן על הסקאלה.
            ללחוץ Enter להמשך.
            """
      ),

        "taskEnd1": (
            """
        סיימת את המטלה הראשונה, 
            נא להמתין לנסיין.
        """
            ),

        "taskEnd2": (
                """
            סיימת את המטלה השניה, 
                נא להמתין לנסיין.
            """

        ),

        "HBC_Start": (
            """
            כעת הספירה תהיה אך ורק של הדופק האמיתי שלך.
        """

        ),

        "CTCT_Start": (
            """
        כעת הספירה תהיה אך ורק של הדופק המדומה שיושמע מהרמקול.
        """

        )
    }


    texts["Rest"] = ("נא לשבת בשקט עד לתחילת החלק הבא.")

    texts['continue_text'] = "נא ללחוץ על מקש Enter כדי להמשיך."
    texts['not_number_input'] = "יש להזין מספרים בלבד."
    texts['task_completion'] = "השלמת את המטלה. תודה על ההשתתפות."



    """ Instructions "Tutorial1" to "Tutorial5" translated and adapted from: 
    Desmedt, O., Luminet, O., and Corneille, O. (2018). 
    The heartbeat counting task largely involves non-interoceptive processes:
    Evidence from both the original and an adapted counting task. 
    Biol. Psychol. 138, 185–188. 
    https://doi.org/10.1016/j.biopsycho.2018.09.004.
    
    The english source version:
        In this task, direct your attention to your heart and the associated physical sensations. 
        You are required to sustain your focus on your heart for various durations and count the number of heartbeats you perceive. 
        Begin silently counting when the heart symbol appears on the screen. 
        When the heart symbol disappears, report the number of heartbeats you are sure you felt. 
        Only report the number of heartbeats you are sure about, without trying to estimate your heart rate. 
        During each trial, keep your eyes open, gaze at the screen and avoid moving. 
        Refrain from guiding your responses by checking your pulse in your wrists or neck. 
        Breathe spontaneously and avoid changing your breathing frequency or holding your breath.
    
    """

    texts["Tutorial1"] = (
        """
        במטלה זו, יש לספור את הדופק האמיתי שלך 
         או דופק מסאונד שמדמה פעימות של לב.
         
          חלונות הזמן של הספירה ישתנה מפעם לפעם.
          
          
    
        """
    )

    texts["Tutorial2"] = (
        """        
        כאשר מופיע איור של לב, ונשמעת המילה "Start" 
        יש להתחיל לספור את הדופק האמיתי.  
        
        כאשר נשמעת המילה "Stop" יש להפסיק את הספירה.
        
        """
    )

    texts[
        "Tutorial3"
    ] = (
    """
         כדי לספור את הדופק, צריך לכוון את הקשב אל הלב שלך
         ולתחושות פיזיות הקשורות לפעילותו.
        כלומר, לחוש אותו מבפנים.
        
        יש לעשות זאת בעיניים פקוחות,
        וללא מגע של היד על העור.
        
        כמו כן, שומרים על נשימה טבעית.
        נמנעים, למשל, מעצירה שלה. 
       
    
        """
    )

    texts["Tutorial4"] = (
        """       
        התהליך זהה לחלוטין כאשר סופרים את הדופק המדומה.
        
        כאשר מופיע ציור של אוזן, ונשמעת המילה "Start" 
        יש להתחיל לספור את הדופק המדומה שיושמע מהרמקול. 

        כאשר נשמעת המילה "Stop" יש להפסיק את הספירה.

        """
        )

    texts["Tutorial5"] = (
        """
        בשני המקרים, צריך להשתדל לא לזוז בכלל,
        שכן ההקלטות בזמן הספירה הן קריטיות.
        
        לאורך כל הניסוי, 
        יש להניח את יד ימין על לוח המספרים בצד הימני של המקלדת
        ואת יד שמאל להניח על הירך.
        
        
        בנוסף, יש למקד את המבט במרכז האיור, מבלי להניע את האישונים.
        """
    )


    texts["Tutorial6"] = (
        """
        לאחר שלב הספירה, מדווחים על כמות הפעימות שספרת.
        לשים לב לכתוב בדיוק כמה ספרת, 
        מבלי להוסיף או להוריד מהמספר הזה בשל אומדנים אחרים שאולי יש לך.
        
        הדיווח נעשה באמצעות המספרים שממוקמים בצד ימין של המקלדת.
        ניתן למחוק מספר שכתבת, באמצעות לחיצה על backspace הרגיל.
        
        כדי לסיים לוחצים על Enter בצד ימין של המקלדת.
        """
    )



    """
    Instructions "Tutorial6" to "Tutorial9" translated and adapted from Legrand's original code (see in the english above)
    """
    texts["Tutorial7"] = (
       """
       בכל פעם אחרי שכותבים את מספר פעימות הלב,
       נאמוד את רמת הבטחון שלך לגבי אותה תשובה.
       
        האומדן יעשה באמצעות סקאלה של בין 100 (ימין עד הסוף) ל- 0 (שמאל עד הסוף).
        
        המשמעות של 100 היא שיש לך בטחון מוחלט שהתשובה שלך נכונה.
        המשמעות של 0 היא שהתשובה שלך הייתה ניחוש.
        
        כדי לבצע שינוי בסקאלה יש להשתמש בספרות 6 (ימינה) או 4 (שמאלה)
        שבצד ימין של המקלדת. 
       
       """
    )

    texts["Tutorial8"] = (
        """
        כעת נבצע אימון קצר.
        """

    )

    texts["Tutorial9"] = (
        """
                מצויין!
                
        לשים לב שמהלך הניסוי עצמו
        כאשר סופרים את פעימות הלב
        יופיע ציור של לב בלבד, ללא הטקסט המצורף.
        """

    )

    texts["Tutorial10"] = (
        """
        סה"כ אורך המטלה כ- 25 דקות.
        
        במידה ויש לך שאלות לגבי הניסוי,
        אפשר לשאול את הנסיין עכשיו.
        
        אחרת, להקיש Enter כדי להתחיל בניסוי.
        """
    )

    if CTCT is True:
        texts["TutorialExtero1"] = (
            """
            במטלה זו, יושמע סאונד שמדמה פעימות לב בחלונות זמן משתנים.
                       
            """
        )

        texts["TutorialExtero2"] = (
            """        
            כאשר מופיע ציור של אוזן, ונשמעת המילה "Start" 
            יש להתחיל לספור את הפעימות המדומות ששומעים.  

            כאשר נשמעת המילה "Stop" יש להפסיק את הספירה.

            בשלב זה, צריך להשתדל במיוחד לא לזוז,
            שכן ההקלטות בו הן קריטיות.
            
            בנוסף, יש למקד את המבט במרכז הציור, 
            מבלי להניע את האישונים.
            """

        )

        # texts[
        #    "Tutorial3"
        # ] = (
        #    'Sometime, you will also encounter this "rest" icon.'
        #    " In this case your task will just be to sit quietly until the next"
        #    " session."
        # )

        texts["TutorialExtero4"] = (
            """
            לאחר מכן,  מדווחים על כמות הפעימות שספרת.

            הדיווח נעשה באמצעות המספרים שממוקמים בצד ימין של המקלדת.
            ניתן למחוק מספר שכתבת, באמצעות לחיצה על backspace הרגיל.
            """
        )

        """
        Instructions "Tutorial6" to "Tutorial9" translated and adapted from Legrand's original code (see in the english above)
        """
        texts["TutorialExtero6"] = (
            """
            בכל פעם אחרי שכותבים את המספר,
            נאמוד את רמת הבטחון שלך לגבי אותה תשובה.

             האומדן יעשה באמצעות סקאלה של בין 100 (ימין עד הסוף) ל- 0 (שמאל עד הסוף).

             המשמעות של 100 היא שיש לך בטחון מוחלט שהתשובה שלך נכונה.
             המשמעות של Enter היא שהתשובה שלך הייתה ניחוש.

             כדי לבצע שינוי בסקאלה יש להשתמש בספרות 6 (ימינה) או 4 (שמאלה)
             שבצד ימין של המקלדת. 

            """
        )

        texts["TutorialExtero7"] = (
            """
            כעת נבצע אימון קצר.
            """

        )

        texts["TutorialExtero8"] = (
            """
                    מצויין!

            לשים לב שמהלך הניסוי עצמו
            כאשר סופרים 
            יופיע ציור של אוזן בלבד, ללא הטקסט המצורף.
            """

        )

        texts["TutorialExtero9"] = (
            """
            סה"כ אורך המטלה כ- 13 דקות.

            במידה ויש לך שאלות לגבי הניסוי,
            אפשר לשאול את הנסיין עכשיו.

            אחרת, להקיש Enter כדי להתחיל בניסוי.
            """
        )

    return texts
