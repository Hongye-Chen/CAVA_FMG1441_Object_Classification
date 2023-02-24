#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2022.2.5),
    on februari 24, 2023, at 11:48
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, parallel, iohub, hardware
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard



# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)
# Store info about the experiment session
psychopyVersion = '2022.2.5'
expName = 'EEG_TargetDetection_Experiment'  # from the Builder filename that created this script
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
}
# --- Show participant info dialog --
dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath='D:\\Users\\Niklas\\eeg_experiment-main\\EEG_TargetDetection_Experiment.py',
    savePickle=True, saveWideText=True,
    dataFileName=filename)
# save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp
frameTolerance = 0.001  # how close to onset before 'same' frame

# Start Code - component code to be run after the window creation

# --- Setup the Window ---
win = visual.Window(
    size=[2560, 1440], fullscr=True, screen=0, 
    winType='pyglet', allowStencil=False,
    monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
    blendMode='avg', useFBO=True, 
    units='height')
win.mouseVisible = False
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess
# --- Setup input devices ---
ioConfig = {}

# Setup eyetracking
ioConfig['eyetracker.hw.sr_research.eyelink.EyeTracker'] = {
    'name': 'tracker',
    'model_name': 'EYELINK 1000 DESKTOP',
    'simulation_mode': False,
    'network_settings': '100.1.1.1',
    'default_native_data_file_name': 'EXPFILE',
    'runtime_settings': {
        'sampling_rate': 500.0,
        'track_eyes': 'RIGHT_EYE',
        'sample_filtering': {
            'sample_filtering': 'FILTER_LEVEL_2',
            'elLiveFiltering': 'FILTER_LEVEL_OFF',
        },
        'vog_settings': {
            'pupil_measure_types': 'PUPIL_DIAMETER',
            'tracking_mode': 'PUPIL_CR_TRACKING',
            'pupil_center_algorithm': 'CENTROID_FIT',
        }
    }
}

# Setup iohub keyboard
ioConfig['Keyboard'] = dict(use_keymap='psychopy')

ioSession = '1'
if 'session' in expInfo:
    ioSession = str(expInfo['session'])
ioServer = io.launchHubServer(window=win, experiment_code='EEG_TargetDetection_Experiment', session_code=ioSession, datastore_name=filename, **ioConfig)
eyetracker = ioServer.getDevice('tracker')

# create a default keyboard (e.g. to check for escape)
defaultKeyboard = keyboard.Keyboard(backend='iohub')

# --- Initialize components for Routine "Initialization" ---
# Run 'Begin Experiment' code from Initialization_code
import pandas as pd

#Set the global file path
#files_path = r"C:\Users\15202\OneDrive\C_\University of Amsterdam\Intern\\"
#
files_path= os.path.join("D:\\", "Users", "Niklas", "eeg_experiment-main")
#os.makedirs(os.path.join(files_path, expInfo['participant']), exist_ok=True)
#images_path = os.path.join("Y:\\", "Projects", "2023_Scholte_FMG1441", "Stimuli")
#newFileName = os.path.join("Y:\\", "Projects", "2023_Scholte_FMG1441", "Data") + u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])

#_thisDir = os.path.dirname(os.path.abspath(r"Y:\Projects\2023_Scholte_FMG1441\Data\pilot"))
#filename = _thisDir + os.sep + u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
#thisExp.dataFileName = filename
#logFile.dataFileName = filename

with open(os.path.join(files_path, "Instruction.md"), 'r') as instruction_file:     #Load the instruction markdown file
    instruction_text = instruction_file.read()          #save the text

eeg_trial_num = 0   #Track with the current trial
cur_triger = 0
images_per_trial = 20
trials_per_block = 5
blocks_per_repetition = 2
total_repetition = 1
total_blocks = blocks_per_repetition * total_repetition
max_trial = trials_per_block * total_blocks #The maximum of trials

orig_image_width = 5496
orig_image_height= 3672
image_w = win.size[0]           #Set the screen width as the width of images
aspect_ratio = orig_image_height/orig_image_width       #Set the aspect ratio of image
image_h = int(image_w * aspect_ratio)     #Set the height of images

len_blank_long = 3          #The length of blank screen before showing any images 
len_blank_short = 0.3       #The length of blank screen after every images

#Get the list of all images' file name
#Images_name_list = os.listdir(images_path)
Images_name_list = os.listdir(os.path.join(files_path, "Stimuli"))

# NOTICE: The Images from the list are not in numerical order, 
# i.e. the third element is "10.tiff" but not "2.tiff"
# In the testing case (and also formal experiment), this would not cause any problem. 
# However, if we need to use these indices to track with specific images, 
# use the command below to sort the list
# (make sure the prefix of file name is number only).

#sorted(Images_name_list, key = lambda item: int(item.split(".")[0]))


etRecord_start = hardware.eyetracker.EyetrackerControl(
    tracker=eyetracker,
    actionType='Start Only'
)

# --- Initialize components for Routine "Introduction" ---
Intro = visual.TextStim(win=win, name='Intro',
    text='Intro',
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
Instruction_content = visual.TextStim(win=win, name='Instruction_content',
    text=instruction_text,
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-1.0);
Instruction_response = keyboard.Keyboard()

# --- Initialize components for Routine "Correctness_Initialization" ---

# --- Initialize components for Routine "Load_current_trial_info" ---

# --- Initialize components for Routine "Preload_and_Blank_Screen" ---
blank_screen = visual.Rect(
    win=win, name='blank_screen',units='pix', 
    width=[image_w, image_h][0], height=[image_w, image_h][1],
    ori=0.0, pos=(0, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='gray', fillColor='gray',
    opacity=None, depth=0.0, interpolate=True)
gray_trigger = parallel.ParallelPort(address='0x4050')
preload_cur_image = clock.StaticPeriod(win=win, screenHz=expInfo['frameRate'], name='preload_cur_image')
Images = visual.ImageStim(
    win=win,
    name='Images', units='pix', 
    image='sin', mask=None, anchor='center',
    ori=0.0, pos=(0, 0), size=[image_w, image_h],
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=True, flipVert=False,
    texRes=128.0, interpolate=True, depth=-3.0)
Image_trigger = parallel.ParallelPort(address='0x4050')
cross = visual.ShapeStim(
    win=win, name='cross', vertices='cross',
    size=(0.01, 0.01),
    ori=0.0, pos=(0, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='red', fillColor='red',
    opacity=None, depth=-5.0, interpolate=True)

# --- Initialize components for Routine "Response" ---
blank_screen_last = visual.Rect(
    win=win, name='blank_screen_last',units='pix', 
    width=[image_w, image_h][0], height=[image_w, image_h][1],
    ori=0.0, pos=(0, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='gray', fillColor='gray',
    opacity=None, depth=0.0, interpolate=True)
corss_last = visual.ShapeStim(
    win=win, name='corss_last', vertices='cross',
    size=(0.01, 0.01),
    ori=0.0, pos=(0, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='red', fillColor='red',
    opacity=None, depth=-1.0, interpolate=True)
last_p_port = parallel.ParallelPort(address='0x4050')
response_image = visual.ImageStim(
    win=win,
    name='response_image', units='pix', 
    image='response.png', mask=None, anchor='center',
    ori=0.0, pos=(0, 0), size=[image_w, image_h],
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-3.0)
key_resp = keyboard.Keyboard()

# --- Initialize components for Routine "Trial_Feedback" ---
trial_feedback = visual.TextStim(win=win, name='trial_feedback',
    text='',
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);

# --- Initialize components for Routine "Feedback" ---
text_3 = visual.TextStim(win=win, name='text_3',
    text='',
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-1.0);

# --- Initialize components for Routine "End_of_Block" ---
text_4 = visual.TextStim(win=win, name='text_4',
    text='',
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-1.0);
Rest = visual.TextStim(win=win, name='Rest',
    text='Please take a rest for 10 seconds',
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-2.0);
PressKey_to_Continue = visual.TextStim(win=win, name='PressKey_to_Continue',
    text='Press ENTER to continue',
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-3.0);
key_resp_2 = keyboard.Keyboard()

# --- Initialize components for Routine "End_of_experiment" ---
etRecord_end = hardware.eyetracker.EyetrackerControl(
    tracker=eyetracker,
    actionType='Stop Only'
)

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine 
# define target for calibration
calibrationTarget = visual.TargetStim(win, 
    name='calibrationTarget',
    radius=0.01, fillColor='', borderColor='black', lineWidth=2.0,
    innerRadius=0.0035, innerFillColor='green', innerBorderColor='black', innerLineWidth=2.0,
    colorSpace='rgb', units=None
)
# define parameters for calibration
calibration = hardware.eyetracker.EyetrackerCalibration(win, 
    eyetracker, calibrationTarget,
    units=None, colorSpace='rgb',
    progressMode='time', targetDur=1.5, expandScale=1.5,
    targetLayout='FIVE_POINTS', randomisePos=True, textColor='white',
    movementAnimation=True, targetDelay=1.0
)
# run calibration
calibration.run()
# clear any keypresses from during calibration so they don't interfere with the experiment
defaultKeyboard.clearEvents()
# the Routine "calibration" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()
# define target for validation
validationTarget = visual.TargetStim(win, 
    name='validationTarget',
    radius=0.01, fillColor='', borderColor='black', lineWidth=2.0,
    innerRadius=0.0035, innerFillColor='green', innerBorderColor='black', innerLineWidth=2.0,
    colorSpace='rgb', units=None
)
# define parameters for validation
validation = iohub.ValidationProcedure(win,
    target=validationTarget,
    gaze_cursor='green', 
    positions='FIVE_POINTS', randomize_positions=True,
    expand_scale=1.5, target_duration=1.5,
    enable_position_animation=True, target_delay=1.0,
    progress_on_key=None, text_color='auto',
    show_results_screen=True, save_results_screen=False,
    color_space='rgb', unit_type=None
)
# run validation
validation.run()
# clear any keypresses from during validation so they don't interfere with the experiment
defaultKeyboard.clearEvents()
# the Routine "validation" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# --- Prepare to start Routine "Initialization" ---
continueRoutine = True
routineForceEnded = False
# update component parameters for each repeat
# keep track of which components have finished
InitializationComponents = [etRecord_start]
for thisComponent in InitializationComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
frameN = -1

# --- Run Routine "Initialization" ---
while continueRoutine:
    # get current time
    t = routineTimer.getTime()
    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    # *etRecord_start* updates
    if etRecord_start.status == NOT_STARTED and t >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        etRecord_start.frameNStart = frameN  # exact frame index
        etRecord_start.tStart = t  # local t and not account for scr refresh
        etRecord_start.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(etRecord_start, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.addData('etRecord_start.started', t)
        etRecord_start.status = STARTED
    if etRecord_start.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > etRecord_start.tStartRefresh + 0-frameTolerance:
            # keep track of stop time/frame for later
            etRecord_start.tStop = t  # not accounting for scr refresh
            etRecord_start.frameNStop = frameN  # exact frame index
            # add timestamp to datafile
            thisExp.addData('etRecord_start.stopped', t)
            etRecord_start.status = FINISHED
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        routineForceEnded = True
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in InitializationComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# --- Ending Routine "Initialization" ---
for thisComponent in InitializationComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# make sure the eyetracker recording stops
if etRecord_start.status != FINISHED:
    etRecord_start.status = FINISHED
# the Routine "Initialization" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of conditions etc
Repetition = data.TrialHandler(nReps=total_repetition, method='sequential', 
    extraInfo=expInfo, originPath=-1,
    trialList=[None],
    seed=None, name='Repetition')
thisExp.addLoop(Repetition)  # add the loop to the experiment
thisRepetition = Repetition.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisRepetition.rgb)
if thisRepetition != None:
    for paramName in thisRepetition:
        exec('{} = thisRepetition[paramName]'.format(paramName))

for thisRepetition in Repetition:
    currentLoop = Repetition
    # abbreviate parameter names if possible (e.g. rgb = thisRepetition.rgb)
    if thisRepetition != None:
        for paramName in thisRepetition:
            exec('{} = thisRepetition[paramName]'.format(paramName))
    
    # --- Prepare to start Routine "Introduction" ---
    continueRoutine = True
    routineForceEnded = False
    # update component parameters for each repeat
    Instruction_response.keys = []
    Instruction_response.rt = []
    _Instruction_response_allKeys = []
    # keep track of which components have finished
    IntroductionComponents = [Intro, Instruction_content, Instruction_response]
    for thisComponent in IntroductionComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Introduction" ---
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Intro* updates
        if Intro.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            Intro.frameNStart = frameN  # exact frame index
            Intro.tStart = t  # local t and not account for scr refresh
            Intro.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Intro, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Intro.started')
            Intro.setAutoDraw(True)
        if Intro.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > Intro.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                Intro.tStop = t  # not accounting for scr refresh
                Intro.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Intro.stopped')
                Intro.setAutoDraw(False)
        
        # *Instruction_content* updates
        if Instruction_content.status == NOT_STARTED and tThisFlip >= 3-frameTolerance:
            # keep track of start time/frame for later
            Instruction_content.frameNStart = frameN  # exact frame index
            Instruction_content.tStart = t  # local t and not account for scr refresh
            Instruction_content.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Instruction_content, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Instruction_content.started')
            Instruction_content.setAutoDraw(True)
        
        # *Instruction_response* updates
        waitOnFlip = False
        if Instruction_response.status == NOT_STARTED and tThisFlip >= 3.0-frameTolerance:
            # keep track of start time/frame for later
            Instruction_response.frameNStart = frameN  # exact frame index
            Instruction_response.tStart = t  # local t and not account for scr refresh
            Instruction_response.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Instruction_response, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Instruction_response.started')
            Instruction_response.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(Instruction_response.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(Instruction_response.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if Instruction_response.status == STARTED and not waitOnFlip:
            theseKeys = Instruction_response.getKeys(keyList=['return','space'], waitRelease=False)
            _Instruction_response_allKeys.extend(theseKeys)
            if len(_Instruction_response_allKeys):
                Instruction_response.keys = _Instruction_response_allKeys[0].name  # just the first key pressed
                Instruction_response.rt = _Instruction_response_allKeys[0].rt
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in IntroductionComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Introduction" ---
    for thisComponent in IntroductionComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # check responses
    if Instruction_response.keys in ['', [], None]:  # No response was made
        Instruction_response.keys = None
    Repetition.addData('Instruction_response.keys',Instruction_response.keys)
    if Instruction_response.keys != None:  # we had a response
        Repetition.addData('Instruction_response.rt', Instruction_response.rt)
    # the Routine "Introduction" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    Blocks = data.TrialHandler(nReps=blocks_per_repetition, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='Blocks')
    thisExp.addLoop(Blocks)  # add the loop to the experiment
    thisBlock = Blocks.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
    if thisBlock != None:
        for paramName in thisBlock:
            exec('{} = thisBlock[paramName]'.format(paramName))
    
    for thisBlock in Blocks:
        currentLoop = Blocks
        # abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
        if thisBlock != None:
            for paramName in thisBlock:
                exec('{} = thisBlock[paramName]'.format(paramName))
        
        # --- Prepare to start Routine "Correctness_Initialization" ---
        continueRoutine = True
        routineForceEnded = False
        # update component parameters for each repeat
        # Run 'Begin Routine' code from Cor_initialization
        num_correct = 0     #For every block, initialize the number of correct answers
        
        # keep track of which components have finished
        Correctness_InitializationComponents = []
        for thisComponent in Correctness_InitializationComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Correctness_Initialization" ---
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Correctness_InitializationComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Correctness_Initialization" ---
        for thisComponent in Correctness_InitializationComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # the Routine "Correctness_Initialization" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        Trials = data.TrialHandler(nReps=trials_per_block, method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=[None],
            seed=None, name='Trials')
        thisExp.addLoop(Trials)  # add the loop to the experiment
        thisTrial = Trials.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                exec('{} = thisTrial[paramName]'.format(paramName))
        
        for thisTrial in Trials:
            currentLoop = Trials
            # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
            if thisTrial != None:
                for paramName in thisTrial:
                    exec('{} = thisTrial[paramName]'.format(paramName))
            
            # --- Prepare to start Routine "Load_current_trial_info" ---
            continueRoutine = True
            routineForceEnded = False
            # update component parameters for each repeat
            # Run 'Begin Routine' code from Load_cur_trial_infos
            #Load the randomized images matrix, 
            #each row of the matrix represents a sequence of images,
            #each element should contain a image number,
            #The code below loads the corresponding row of trial from the matrix,
            #please change the path below
            cur_sequence = pd.read_csv(os.path.join(files_path, "Randomized matrix.csv"), header = None, skiprows = list(range(0,eeg_trial_num)) + list(range(eeg_trial_num+1,max_trial)) )
            cur_sequence_images = [Images_name_list[cur_sequence.iloc[0, i]] for i in range(images_per_trial) ]     #Use the indexs to get the images' name
            cur_sequence_images = [ "Stimuli\\" + i for i in cur_sequence_images]
            #cur_sequence_images = [ os.path.join(images_path, i) for i in cur_sequence_images]
            
            #Load the csv file which contain the correct answers for each trial, 
            #each row contains one answer for a trial, the value should be either 'y' or 'n',
            #The code below loads the corresponding row of trial from the file,
            #please change the path below
            cur_correct = pd.read_csv(os.path.join(files_path, "Correct_answers.csv"), header = None, skiprows = list(range(0,eeg_trial_num)) + list(range(eeg_trial_num+1,max_trial)) )
            cur_correct = cur_correct[0][0]     #Get the correct answer
            eeg_trial_num += 1      #Increase the trial index
            
            cur_trial_data = {'current_images_list' : cur_sequence_images, 'blank_len' : [len_blank_long] + [len_blank_short] * (images_per_trial-1) }     
            cur_sequence_images = pd.DataFrame(cur_trial_data)      #Convert the sequence into DataFrame
            
            #Save the current images list into the loop file
            cur_sequence_images.to_csv(os.path.join(files_path, "loopTemplate.csv"),index=False)
            
            #import PIL.Image
            #current_images = []
            #for image_index in range(20):
            #    img = Image.open(filenames[image_index])
            #    current_images.append(img)
            
            gray_trigger_index = 251
            # keep track of which components have finished
            Load_current_trial_infoComponents = []
            for thisComponent in Load_current_trial_infoComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "Load_current_trial_info" ---
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in Load_current_trial_infoComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "Load_current_trial_info" ---
            for thisComponent in Load_current_trial_infoComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # the Routine "Load_current_trial_info" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # set up handler to look after randomisation of conditions etc
            Sequences = data.TrialHandler(nReps=1.0, method='sequential', 
                extraInfo=expInfo, originPath=-1,
                trialList=data.importConditions('loopTemplate.csv'),
                seed=None, name='Sequences')
            thisExp.addLoop(Sequences)  # add the loop to the experiment
            thisSequence = Sequences.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisSequence.rgb)
            if thisSequence != None:
                for paramName in thisSequence:
                    exec('{} = thisSequence[paramName]'.format(paramName))
            
            for thisSequence in Sequences:
                currentLoop = Sequences
                # abbreviate parameter names if possible (e.g. rgb = thisSequence.rgb)
                if thisSequence != None:
                    for paramName in thisSequence:
                        exec('{} = thisSequence[paramName]'.format(paramName))
                
                # --- Prepare to start Routine "Preload_and_Blank_Screen" ---
                continueRoutine = True
                routineForceEnded = False
                # update component parameters for each repeat
                # keep track of which components have finished
                Preload_and_Blank_ScreenComponents = [blank_screen, gray_trigger, preload_cur_image, Images, Image_trigger, cross]
                for thisComponent in Preload_and_Blank_ScreenComponents:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "Preload_and_Blank_Screen" ---
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *blank_screen* updates
                    if blank_screen.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        blank_screen.frameNStart = frameN  # exact frame index
                        blank_screen.tStart = t  # local t and not account for scr refresh
                        blank_screen.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(blank_screen, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'blank_screen.started')
                        blank_screen.setAutoDraw(True)
                    if blank_screen.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > blank_screen.tStartRefresh + blank_len-frameTolerance:
                            # keep track of stop time/frame for later
                            blank_screen.tStop = t  # not accounting for scr refresh
                            blank_screen.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'blank_screen.stopped')
                            blank_screen.setAutoDraw(False)
                    # *gray_trigger* updates
                    if gray_trigger.status == NOT_STARTED and blank_screen.status == STARTED:
                        # keep track of start time/frame for later
                        gray_trigger.frameNStart = frameN  # exact frame index
                        gray_trigger.tStart = t  # local t and not account for scr refresh
                        gray_trigger.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(gray_trigger, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'gray_trigger.started')
                        gray_trigger.status = STARTED
                        win.callOnFlip(gray_trigger.setData, int(gray_trigger_index))
                    if gray_trigger.status == STARTED:
                        if frameN >= (gray_trigger.frameNStart + 1):
                            # keep track of stop time/frame for later
                            gray_trigger.tStop = t  # not accounting for scr refresh
                            gray_trigger.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'gray_trigger.stopped')
                            gray_trigger.status = FINISHED
                            win.callOnFlip(gray_trigger.setData, int())
                    
                    # *Images* updates
                    if Images.status == NOT_STARTED and preload_cur_image.status==FINISHED:
                        # keep track of start time/frame for later
                        Images.frameNStart = frameN  # exact frame index
                        Images.tStart = t  # local t and not account for scr refresh
                        Images.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(Images, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'Images.started')
                        Images.setAutoDraw(True)
                    if Images.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > Images.tStartRefresh + 0.1-frameTolerance:
                            # keep track of stop time/frame for later
                            Images.tStop = t  # not accounting for scr refresh
                            Images.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'Images.stopped')
                            Images.setAutoDraw(False)
                    # *Image_trigger* updates
                    if Image_trigger.status == NOT_STARTED and Images.status == STARTED:
                        # keep track of start time/frame for later
                        Image_trigger.frameNStart = frameN  # exact frame index
                        Image_trigger.tStart = t  # local t and not account for scr refresh
                        Image_trigger.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(Image_trigger, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'Image_trigger.started')
                        Image_trigger.status = STARTED
                        win.callOnFlip(Image_trigger.setData, int(cur_triger))
                    if Image_trigger.status == STARTED:
                        if frameN >= (Image_trigger.frameNStart + 1):
                            # keep track of stop time/frame for later
                            Image_trigger.tStop = t  # not accounting for scr refresh
                            Image_trigger.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'Image_trigger.stopped')
                            Image_trigger.status = FINISHED
                            win.callOnFlip(Image_trigger.setData, int())
                    
                    # *cross* updates
                    if cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        cross.frameNStart = frameN  # exact frame index
                        cross.tStart = t  # local t and not account for scr refresh
                        cross.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(cross, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'cross.started')
                        cross.setAutoDraw(True)
                    if cross.status == STARTED:
                        if bool(Images.status==FINISHED):
                            # keep track of stop time/frame for later
                            cross.tStop = t  # not accounting for scr refresh
                            cross.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'cross.stopped')
                            cross.setAutoDraw(False)
                    # *preload_cur_image* period
                    if preload_cur_image.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                        # keep track of start time/frame for later
                        preload_cur_image.frameNStart = frameN  # exact frame index
                        preload_cur_image.tStart = t  # local t and not account for scr refresh
                        preload_cur_image.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(preload_cur_image, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'preload_cur_image.started')
                        preload_cur_image.start(blank_len)
                    elif preload_cur_image.status == STARTED:  # one frame should pass before updating params and completing
                        # Updating other components during *preload_cur_image*
                        Images.setImage(current_images_list)
                        # Component updates done
                        preload_cur_image.complete()  # finish the static period
                        preload_cur_image.tStop = preload_cur_image.tStart + blank_len  # record stop time
                    
                    # check for quit (typically the Esc key)
                    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                        core.quit()
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in Preload_and_Blank_ScreenComponents:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "Preload_and_Blank_Screen" ---
                for thisComponent in Preload_and_Blank_ScreenComponents:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                if gray_trigger.status == STARTED:
                    win.callOnFlip(gray_trigger.setData, int())
                if Image_trigger.status == STARTED:
                    win.callOnFlip(Image_trigger.setData, int())
                # Run 'End Routine' code from trigger_update
                gray_trigger_index = 253
                if cur_triger == 255:
                    cur_triger = 0
                else:
                    cur_triger += 1
                
                # the Routine "Preload_and_Blank_Screen" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                thisExp.nextEntry()
                
            # completed 1.0 repeats of 'Sequences'
            
            
            # --- Prepare to start Routine "Response" ---
            continueRoutine = True
            routineForceEnded = False
            # update component parameters for each repeat
            key_resp.keys = []
            key_resp.rt = []
            _key_resp_allKeys = []
            # keep track of which components have finished
            ResponseComponents = [blank_screen_last, corss_last, last_p_port, response_image, key_resp]
            for thisComponent in ResponseComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "Response" ---
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *blank_screen_last* updates
                if blank_screen_last.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    blank_screen_last.frameNStart = frameN  # exact frame index
                    blank_screen_last.tStart = t  # local t and not account for scr refresh
                    blank_screen_last.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(blank_screen_last, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'blank_screen_last.started')
                    blank_screen_last.setAutoDraw(True)
                if blank_screen_last.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > blank_screen_last.tStartRefresh + len_blank_short-frameTolerance:
                        # keep track of stop time/frame for later
                        blank_screen_last.tStop = t  # not accounting for scr refresh
                        blank_screen_last.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'blank_screen_last.stopped')
                        blank_screen_last.setAutoDraw(False)
                
                # *corss_last* updates
                if corss_last.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    corss_last.frameNStart = frameN  # exact frame index
                    corss_last.tStart = t  # local t and not account for scr refresh
                    corss_last.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(corss_last, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'corss_last.started')
                    corss_last.setAutoDraw(True)
                if corss_last.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > corss_last.tStartRefresh + len_blank_short-frameTolerance:
                        # keep track of stop time/frame for later
                        corss_last.tStop = t  # not accounting for scr refresh
                        corss_last.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'corss_last.stopped')
                        corss_last.setAutoDraw(False)
                # *last_p_port* updates
                if last_p_port.status == NOT_STARTED and blank_screen_last.status == STARTED:
                    # keep track of start time/frame for later
                    last_p_port.frameNStart = frameN  # exact frame index
                    last_p_port.tStart = t  # local t and not account for scr refresh
                    last_p_port.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(last_p_port, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'last_p_port.started')
                    last_p_port.status = STARTED
                    win.callOnFlip(last_p_port.setData, int(gray_trigger_index))
                if last_p_port.status == STARTED:
                    if frameN >= (last_p_port.frameNStart + 1):
                        # keep track of stop time/frame for later
                        last_p_port.tStop = t  # not accounting for scr refresh
                        last_p_port.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'last_p_port.stopped')
                        last_p_port.status = FINISHED
                        win.callOnFlip(last_p_port.setData, int())
                
                # *response_image* updates
                if response_image.status == NOT_STARTED and blank_screen_last.status == FINISHED:
                    # keep track of start time/frame for later
                    response_image.frameNStart = frameN  # exact frame index
                    response_image.tStart = t  # local t and not account for scr refresh
                    response_image.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(response_image, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'response_image.started')
                    response_image.setAutoDraw(True)
                if response_image.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > response_image.tStartRefresh + 3-frameTolerance:
                        # keep track of stop time/frame for later
                        response_image.tStop = t  # not accounting for scr refresh
                        response_image.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'response_image.stopped')
                        response_image.setAutoDraw(False)
                
                # *key_resp* updates
                waitOnFlip = False
                if key_resp.status == NOT_STARTED and blank_screen_last.status == FINISHED:
                    # keep track of start time/frame for later
                    key_resp.frameNStart = frameN  # exact frame index
                    key_resp.tStart = t  # local t and not account for scr refresh
                    key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp.started')
                    key_resp.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_resp.status == STARTED:
                    # is it time to stop? (based on local clock)
                    if tThisFlip > 3-frameTolerance:
                        # keep track of stop time/frame for later
                        key_resp.tStop = t  # not accounting for scr refresh
                        key_resp.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'key_resp.stopped')
                        key_resp.status = FINISHED
                if key_resp.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp.getKeys(keyList=['l', 'k'], waitRelease=False)
                    _key_resp_allKeys.extend(theseKeys)
                    if len(_key_resp_allKeys):
                        key_resp.keys = _key_resp_allKeys[0].name  # just the first key pressed
                        key_resp.rt = _key_resp_allKeys[0].rt
                        # was this correct?
                        if (key_resp.keys == str(cur_correct)) or (key_resp.keys == cur_correct):
                            key_resp.corr = 1
                        else:
                            key_resp.corr = 0
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in ResponseComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "Response" ---
            for thisComponent in ResponseComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            if last_p_port.status == STARTED:
                win.callOnFlip(last_p_port.setData, int())
            # check responses
            if key_resp.keys in ['', [], None]:  # No response was made
                key_resp.keys = None
                # was no response the correct answer?!
                if str(cur_correct).lower() == 'none':
                   key_resp.corr = 1;  # correct non-response
                else:
                   key_resp.corr = 0;  # failed to respond (incorrectly)
            # store data for Trials (TrialHandler)
            Trials.addData('key_resp.keys',key_resp.keys)
            Trials.addData('key_resp.corr', key_resp.corr)
            if key_resp.keys != None:  # we had a response
                Trials.addData('key_resp.rt', key_resp.rt)
            # Run 'End Routine' code from Calculate_cor
            trial_correct = "WRONG!"
            
            if key_resp.corr:
                num_correct += 1        #Record how many times the participants reponse correctly
                trial_correct = "CORRECT!"
            else:
                trial_correct = "WRONG!"
            # the Routine "Response" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "Trial_Feedback" ---
            continueRoutine = True
            routineForceEnded = False
            # update component parameters for each repeat
            trial_feedback.setText(trial_correct)
            # keep track of which components have finished
            Trial_FeedbackComponents = [trial_feedback]
            for thisComponent in Trial_FeedbackComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "Trial_Feedback" ---
            while continueRoutine and routineTimer.getTime() < 0.4:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *trial_feedback* updates
                if trial_feedback.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    trial_feedback.frameNStart = frameN  # exact frame index
                    trial_feedback.tStart = t  # local t and not account for scr refresh
                    trial_feedback.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(trial_feedback, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'trial_feedback.started')
                    trial_feedback.setAutoDraw(True)
                if trial_feedback.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > trial_feedback.tStartRefresh + 0.4-frameTolerance:
                        # keep track of stop time/frame for later
                        trial_feedback.tStop = t  # not accounting for scr refresh
                        trial_feedback.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'trial_feedback.stopped')
                        trial_feedback.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in Trial_FeedbackComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "Trial_Feedback" ---
            for thisComponent in Trial_FeedbackComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.400000)
            thisExp.nextEntry()
            
        # completed trials_per_block repeats of 'Trials'
        
        
        # --- Prepare to start Routine "Feedback" ---
        continueRoutine = True
        routineForceEnded = False
        # update component parameters for each repeat
        # Run 'Begin Routine' code from accuracy_rate
        num_correct = round( (num_correct / trials_per_block) * 100 , 2)       #Calculate accuracy rate
        accuracy_rate = str("Your accuracy rate is ") + str(num_correct) + str("%")
        text_3.setText(accuracy_rate)
        # keep track of which components have finished
        FeedbackComponents = [text_3]
        for thisComponent in FeedbackComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Feedback" ---
        while continueRoutine and routineTimer.getTime() < 4.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_3* updates
            if text_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_3.frameNStart = frameN  # exact frame index
                text_3.tStart = t  # local t and not account for scr refresh
                text_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_3.started')
                text_3.setAutoDraw(True)
            if text_3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_3.tStartRefresh + 4-frameTolerance:
                    # keep track of stop time/frame for later
                    text_3.tStop = t  # not accounting for scr refresh
                    text_3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_3.stopped')
                    text_3.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in FeedbackComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Feedback" ---
        for thisComponent in FeedbackComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-4.000000)
        
        # --- Prepare to start Routine "End_of_Block" ---
        continueRoutine = True
        routineForceEnded = False
        # update component parameters for each repeat
        # Run 'Begin Routine' code from block_end
        cur_block = int(eeg_trial_num / trials_per_block)
        block_end_text = "This the end of block " + str(cur_block)
        text_4.setText(block_end_text)
        key_resp_2.keys = []
        key_resp_2.rt = []
        _key_resp_2_allKeys = []
        # keep track of which components have finished
        End_of_BlockComponents = [text_4, Rest, PressKey_to_Continue, key_resp_2]
        for thisComponent in End_of_BlockComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "End_of_Block" ---
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_4* updates
            if text_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_4.frameNStart = frameN  # exact frame index
                text_4.tStart = t  # local t and not account for scr refresh
                text_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_4.started')
                text_4.setAutoDraw(True)
            if text_4.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_4.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    text_4.tStop = t  # not accounting for scr refresh
                    text_4.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_4.stopped')
                    text_4.setAutoDraw(False)
            
            # *Rest* updates
            if Rest.status == NOT_STARTED and tThisFlip >= 3-frameTolerance:
                # keep track of start time/frame for later
                Rest.frameNStart = frameN  # exact frame index
                Rest.tStart = t  # local t and not account for scr refresh
                Rest.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Rest, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Rest.started')
                Rest.setAutoDraw(True)
            if Rest.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Rest.tStartRefresh + 10-frameTolerance:
                    # keep track of stop time/frame for later
                    Rest.tStop = t  # not accounting for scr refresh
                    Rest.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Rest.stopped')
                    Rest.setAutoDraw(False)
            
            # *PressKey_to_Continue* updates
            if PressKey_to_Continue.status == NOT_STARTED and tThisFlip >= 13-frameTolerance:
                # keep track of start time/frame for later
                PressKey_to_Continue.frameNStart = frameN  # exact frame index
                PressKey_to_Continue.tStart = t  # local t and not account for scr refresh
                PressKey_to_Continue.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(PressKey_to_Continue, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'PressKey_to_Continue.started')
                PressKey_to_Continue.setAutoDraw(True)
            
            # *key_resp_2* updates
            waitOnFlip = False
            if key_resp_2.status == NOT_STARTED and tThisFlip >= 13-frameTolerance:
                # keep track of start time/frame for later
                key_resp_2.frameNStart = frameN  # exact frame index
                key_resp_2.tStart = t  # local t and not account for scr refresh
                key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_2.started')
                key_resp_2.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_2.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_2.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_2.getKeys(keyList=['return'], waitRelease=False)
                _key_resp_2_allKeys.extend(theseKeys)
                if len(_key_resp_2_allKeys):
                    key_resp_2.keys = _key_resp_2_allKeys[0].name  # just the first key pressed
                    key_resp_2.rt = _key_resp_2_allKeys[0].rt
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in End_of_BlockComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "End_of_Block" ---
        for thisComponent in End_of_BlockComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # check responses
        if key_resp_2.keys in ['', [], None]:  # No response was made
            key_resp_2.keys = None
        Blocks.addData('key_resp_2.keys',key_resp_2.keys)
        if key_resp_2.keys != None:  # we had a response
            Blocks.addData('key_resp_2.rt', key_resp_2.rt)
        # the Routine "End_of_Block" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed blocks_per_repetition repeats of 'Blocks'
    
    thisExp.nextEntry()
    
# completed total_repetition repeats of 'Repetition'


# --- Prepare to start Routine "End_of_experiment" ---
continueRoutine = True
routineForceEnded = False
# update component parameters for each repeat
# keep track of which components have finished
End_of_experimentComponents = [etRecord_end]
for thisComponent in End_of_experimentComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
frameN = -1

# --- Run Routine "End_of_experiment" ---
while continueRoutine:
    # get current time
    t = routineTimer.getTime()
    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    # *etRecord_end* updates
    if etRecord_end.status == NOT_STARTED and t >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        etRecord_end.frameNStart = frameN  # exact frame index
        etRecord_end.tStart = t  # local t and not account for scr refresh
        etRecord_end.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(etRecord_end, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.addData('etRecord_end.started', t)
        etRecord_end.status = STARTED
    if etRecord_end.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > etRecord_end.tStartRefresh + 1.0-frameTolerance:
            # keep track of stop time/frame for later
            etRecord_end.tStop = t  # not accounting for scr refresh
            etRecord_end.frameNStop = frameN  # exact frame index
            # add timestamp to datafile
            thisExp.addData('etRecord_end.stopped', t)
            etRecord_end.status = FINISHED
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        routineForceEnded = True
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in End_of_experimentComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# --- Ending Routine "End_of_experiment" ---
for thisComponent in End_of_experimentComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# make sure the eyetracker recording stops
if etRecord_end.status != FINISHED:
    etRecord_end.status = FINISHED

# --- End experiment ---
# Flip one final time so any remaining win.callOnFlip() 
# and win.timeOnFlip() tasks get executed before quitting
win.flip()

# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename+'.csv', delim='comma')
thisExp.saveAsPickle(filename)
logging.flush()
# make sure everything is closed down
if eyetracker:
    eyetracker.setConnectionState(False)
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()
