#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2022.2.5),
    on maart 17, 2023, at 10:45
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, iohub, hardware, parallel
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
expName = 'FMG1441_Object_Classification_EEG'  # from the Builder filename that created this script
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
    originPath='D:\\Users\\Niklas\\eeg_experiment-main\\FMG1441_Object_Classification_EEG.py',
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
    size=[2560, 1440], fullscr=True, screen=2, 
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
ioServer = io.launchHubServer(window=win, experiment_code='FMG1441_Object_Classification_EEG', session_code=ioSession, datastore_name=filename, **ioConfig)
eyetracker = ioServer.getDevice('tracker')

# create a default keyboard (e.g. to check for escape)
defaultKeyboard = keyboard.Keyboard(backend='iohub')

# --- Initialize components for Routine "Initialization" ---
# Run 'Begin Experiment' code from Initialization_code
import pandas as pd
from copy import deepcopy
import random
import sys
import gc

#Set the global file path
#files_path = r"C:\Users\15202\OneDrive\C_\University of Amsterdam\Intern\\"
#
files_path= os.path.join("D:\\", "Users", "Niklas", "eeg_experiment-main")
matrix_name = expInfo['participant'] + "_randomized_matrix_702.csv"
save_to_data = pd.read_csv(os.path.join(files_path, "Subject_matrix", matrix_name), header = None)
save_to_data.to_csv(os.path.join(files_path, "data", expInfo['date']+"_"+matrix_name),index=False, header = None)
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
training_trial = 0

cur_triger = 0
images_per_trial = 20
trials_per_block = 30
blocks_per_repetition = 8
total_repetition = 1
total_blocks = blocks_per_repetition * total_repetition
max_trial = trials_per_block * total_blocks #The maximum of trials
Training_Repetition = 9999
number_of_sub = len(os.listdir(os.path.join(files_path, "Subject_matrix"))) - 1    #Get the amount of subjets

orig_image_width = 2155
orig_image_height= 1440
image_h = win.size[1]           #Set the screen width as the width of images
aspect_ratio = orig_image_width/orig_image_height       #Set the aspect ratio of image
image_w = int(image_h * aspect_ratio)     #Set the height of images

#len_blank_long = 3          #The length of blank screen before showing any images 
#len_blank_short = 0.3       #The length of blank screen after every images

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

num_correct = 0     #initialize the number of correct answers

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
    text='In this experiment, you will be rapidly presented with sequences of images.\n\nMost sequences contain only images of outdoor scenes, while other sequences contain outdoor scenes and one indoor scene.\n\nPlease fixate the red cross in the center the entire time and try not to blink during the sequence. \n\nPress SPACE to continue',
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=1.7, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-1.0);
Instruction_response = keyboard.Keyboard()
Instruction_content_2 = visual.TextStim(win=win, name='Instruction_content_2',
    text='At the end of each sequence, you will be asked whether there was an indoor scene present in the current sequence.\n\nUse your RIGHT hand to press the BLUE button for YES, and the GREEN button for NO.\n\nPress SPACE to continue',
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=1.6, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-3.0);
Instruction_response_2 = keyboard.Keyboard()
Instruction_content_3 = visual.TextStim(win=win, name='Instruction_content_3',
    text='Press SPACE to start with some practice trials before starting the experiment.',
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=1.6, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-5.0);
Instruction_response_3 = keyboard.Keyboard()

# --- Initialize components for Routine "Training_Preload_info" ---
Training_Long_gray_screen = visual.Rect(
    win=win, name='Training_Long_gray_screen',units='pix', 
    width=[image_w, image_h][0], height=[image_w, image_h][1],
    ori=0.0, pos=(0, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='gray', fillColor='gray',
    opacity=None, depth=-1.0, interpolate=True)
Training_Long_Cross = visual.ShapeStim(
    win=win, name='Training_Long_Cross', vertices='cross',
    size=(0.01, 0.01),
    ori=0.0, pos=(0, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='red', fillColor='red',
    opacity=None, depth=-2.0, interpolate=True)

# --- Initialize components for Routine "Training" ---
Training_Images = visual.ImageStim(
    win=win,
    name='Training_Images', units='pix', 
    image=None, mask=None, anchor='center',
    ori=0.0, pos=(0, 0), size=[image_w, image_h],
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=True, flipVert=False,
    texRes=128.0, interpolate=True, depth=0.0)
Training_GrayScreen = visual.Rect(
    win=win, name='Training_GrayScreen',units='pix', 
    width=[image_w, image_h][0], height=[image_w, image_h][1],
    ori=0.0, pos=(0, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='gray', fillColor='gray',
    opacity=None, depth=-1.0, interpolate=True)
Training_Cross = visual.ShapeStim(
    win=win, name='Training_Cross', vertices='cross',
    size=(0.01, 0.01),
    ori=0.0, pos=(0, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='red', fillColor='red',
    opacity=None, depth=-2.0, interpolate=True)

# --- Initialize components for Routine "Training_Responses" ---
Training_Responses_Text = visual.TextStim(win=win, name='Training_Responses_Text',
    text='Target?\n\n                  \nYes?:                 No?:',
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
Training_Blue_Circle = visual.ShapeStim(
    win=win, name='Training_Blue_Circle',
    size=(0.1, 0.1), vertices='circle',
    ori=0.0, pos=(-0.04, -0.08), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='blue', fillColor=[-1.0000, -1.0000, 1.0000],
    opacity=None, depth=-1.0, interpolate=True)
Training_Green_Circle = visual.ShapeStim(
    win=win, name='Training_Green_Circle',
    size=(0.1, 0.1), vertices='circle',
    ori=0.0, pos=(0.34, -0.08), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor=[-0.0039, 1.0000, -1.0000], fillColor=[-0.0039, 1.0000, -1.0000],
    opacity=None, depth=-2.0, interpolate=True)
Training_Key_Response = keyboard.Keyboard()

# --- Initialize components for Routine "Training_Feedback" ---
Training_Trial_Feedback = visual.TextStim(win=win, name='Training_Trial_Feedback',
    text='',
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);

# --- Initialize components for Routine "End_of_Training_Start_Formal_Experiment" ---
end_of_training = visual.TextStim(win=win, name='end_of_training',
    text='This is the end of practice trials\n\nPress R to retry the practice trials\n\nPress SPACE to go into the formal experiment',
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
Training_End = keyboard.Keyboard()

# --- Initialize components for Routine "Load_current_trial_info" ---
Long_gray_screen = visual.Rect(
    win=win, name='Long_gray_screen',units='pix', 
    width=[image_w, image_h][0], height=[image_w, image_h][1],
    ori=0.0, pos=(0, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='gray', fillColor='gray',
    opacity=None, depth=-1.0, interpolate=True)
First_p_port_2 = parallel.ParallelPort(address='0x4050')
cross_2 = visual.ShapeStim(
    win=win, name='cross_2', vertices='cross',
    size=(0.01, 0.01),
    ori=0.0, pos=(0, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='red', fillColor='red',
    opacity=None, depth=-3.0, interpolate=True)

# --- Initialize components for Routine "Preload_and_Blank_Screen" ---
Images = visual.ImageStim(
    win=win,
    name='Images', units='pix', 
    image=None, mask=None, anchor='center',
    ori=0.0, pos=(0, 0), size=[image_w, image_h],
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=True, flipVert=False,
    texRes=128.0, interpolate=True, depth=0.0)
Image_trigger = parallel.ParallelPort(address='0x4050')
blank_screen = visual.Rect(
    win=win, name='blank_screen',units='pix', 
    width=[image_w, image_h][0], height=[image_w, image_h][1],
    ori=0.0, pos=(0, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='gray', fillColor='gray',
    opacity=None, depth=-2.0, interpolate=True)
gray_trigger = parallel.ParallelPort(address='0x4050')
cross = visual.ShapeStim(
    win=win, name='cross', vertices='cross',
    size=(0.01, 0.01),
    ori=0.0, pos=(0, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='red', fillColor='red',
    opacity=None, depth=-4.0, interpolate=True)

# --- Initialize components for Routine "Response" ---
response_text = visual.TextStim(win=win, name='response_text',
    text='Target?\n\n                  \nYes?:                 No?:',
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
blue_circle = visual.ShapeStim(
    win=win, name='blue_circle',
    size=(0.1, 0.1), vertices='circle',
    ori=0.0, pos=(-0.04, -0.08), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='blue', fillColor=[-1.0000, -1.0000, 1.0000],
    opacity=None, depth=-1.0, interpolate=True)
Green_circle = visual.ShapeStim(
    win=win, name='Green_circle',
    size=(0.1, 0.1), vertices='circle',
    ori=0.0, pos=(0.34, -0.08), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor=[-0.0039, 1.0000, -1.0000], fillColor=[-0.0039, 1.0000, -1.0000],
    opacity=None, depth=-2.0, interpolate=True)
key_resp = keyboard.Keyboard()

# --- Initialize components for Routine "Trial_Feedback" ---
trial_feedback = visual.TextStim(win=win, name='trial_feedback',
    text='',
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);

# --- Initialize components for Routine "Block_Feedback" ---
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
    text='Please take a rest for at leat 45 seconds',
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-2.0);
PressKey_to_Continue = visual.TextStim(win=win, name='PressKey_to_Continue',
    text='Please take a rest for at leat 45 seconds\n\n\nPress SPACE to continue',
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
text = visual.TextStim(win=win, name='text',
    text='This is the end of experiment\n\nThank you so much for participating!',
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-1.0);

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine 

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

# --- Prepare to start Routine "Introduction" ---
continueRoutine = True
routineForceEnded = False
# update component parameters for each repeat
Instruction_response.keys = []
Instruction_response.rt = []
_Instruction_response_allKeys = []
Instruction_response_2.keys = []
Instruction_response_2.rt = []
_Instruction_response_2_allKeys = []
Instruction_response_3.keys = []
Instruction_response_3.rt = []
_Instruction_response_3_allKeys = []
# keep track of which components have finished
IntroductionComponents = [Intro, Instruction_content, Instruction_response, Instruction_content_2, Instruction_response_2, Instruction_content_3, Instruction_response_3]
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
    if Instruction_content.status == STARTED:
        if bool(Instruction_response.status == FINISHED):
            # keep track of stop time/frame for later
            Instruction_content.tStop = t  # not accounting for scr refresh
            Instruction_content.frameNStop = frameN  # exact frame index
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Instruction_content.stopped')
            Instruction_content.setAutoDraw(False)
    
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
    if Instruction_response.status == STARTED:
        if bool(Instruction_response.keys != []):
            # keep track of stop time/frame for later
            Instruction_response.tStop = t  # not accounting for scr refresh
            Instruction_response.frameNStop = frameN  # exact frame index
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Instruction_response.stopped')
            Instruction_response.status = FINISHED
    if Instruction_response.status == STARTED and not waitOnFlip:
        theseKeys = Instruction_response.getKeys(keyList=['space'], waitRelease=False)
        _Instruction_response_allKeys.extend(theseKeys)
        if len(_Instruction_response_allKeys):
            Instruction_response.keys = _Instruction_response_allKeys[0].name  # just the first key pressed
            Instruction_response.rt = _Instruction_response_allKeys[0].rt
    
    # *Instruction_content_2* updates
    if Instruction_content_2.status == NOT_STARTED and Instruction_response.status == FINISHED:
        # keep track of start time/frame for later
        Instruction_content_2.frameNStart = frameN  # exact frame index
        Instruction_content_2.tStart = t  # local t and not account for scr refresh
        Instruction_content_2.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(Instruction_content_2, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'Instruction_content_2.started')
        Instruction_content_2.setAutoDraw(True)
    if Instruction_content_2.status == STARTED:
        if bool(Instruction_response_2.status == FINISHED):
            # keep track of stop time/frame for later
            Instruction_content_2.tStop = t  # not accounting for scr refresh
            Instruction_content_2.frameNStop = frameN  # exact frame index
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Instruction_content_2.stopped')
            Instruction_content_2.setAutoDraw(False)
    
    # *Instruction_response_2* updates
    waitOnFlip = False
    if Instruction_response_2.status == NOT_STARTED and Instruction_content_2.status == STARTED:
        # keep track of start time/frame for later
        Instruction_response_2.frameNStart = frameN  # exact frame index
        Instruction_response_2.tStart = t  # local t and not account for scr refresh
        Instruction_response_2.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(Instruction_response_2, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'Instruction_response_2.started')
        Instruction_response_2.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(Instruction_response_2.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(Instruction_response_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if Instruction_response_2.status == STARTED:
        if bool(Instruction_response_2.keys != []):
            # keep track of stop time/frame for later
            Instruction_response_2.tStop = t  # not accounting for scr refresh
            Instruction_response_2.frameNStop = frameN  # exact frame index
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Instruction_response_2.stopped')
            Instruction_response_2.status = FINISHED
    if Instruction_response_2.status == STARTED and not waitOnFlip:
        theseKeys = Instruction_response_2.getKeys(keyList=['space'], waitRelease=False)
        _Instruction_response_2_allKeys.extend(theseKeys)
        if len(_Instruction_response_2_allKeys):
            Instruction_response_2.keys = _Instruction_response_2_allKeys[0].name  # just the first key pressed
            Instruction_response_2.rt = _Instruction_response_2_allKeys[0].rt
    
    # *Instruction_content_3* updates
    if Instruction_content_3.status == NOT_STARTED and Instruction_response_2.status == FINISHED:
        # keep track of start time/frame for later
        Instruction_content_3.frameNStart = frameN  # exact frame index
        Instruction_content_3.tStart = t  # local t and not account for scr refresh
        Instruction_content_3.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(Instruction_content_3, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'Instruction_content_3.started')
        Instruction_content_3.setAutoDraw(True)
    if Instruction_content_3.status == STARTED:
        if bool(Instruction_response_3.status == FINISHED):
            # keep track of stop time/frame for later
            Instruction_content_3.tStop = t  # not accounting for scr refresh
            Instruction_content_3.frameNStop = frameN  # exact frame index
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Instruction_content_3.stopped')
            Instruction_content_3.setAutoDraw(False)
    
    # *Instruction_response_3* updates
    waitOnFlip = False
    if Instruction_response_3.status == NOT_STARTED and Instruction_content_3.status == STARTED:
        # keep track of start time/frame for later
        Instruction_response_3.frameNStart = frameN  # exact frame index
        Instruction_response_3.tStart = t  # local t and not account for scr refresh
        Instruction_response_3.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(Instruction_response_3, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'Instruction_response_3.started')
        Instruction_response_3.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(Instruction_response_3.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(Instruction_response_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if Instruction_response_3.status == STARTED:
        if bool(Instruction_response_3.keys != []):
            # keep track of stop time/frame for later
            Instruction_response_3.tStop = t  # not accounting for scr refresh
            Instruction_response_3.frameNStop = frameN  # exact frame index
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Instruction_response_3.stopped')
            Instruction_response_3.status = FINISHED
    if Instruction_response_3.status == STARTED and not waitOnFlip:
        theseKeys = Instruction_response_3.getKeys(keyList=['space'], waitRelease=False)
        _Instruction_response_3_allKeys.extend(theseKeys)
        if len(_Instruction_response_3_allKeys):
            Instruction_response_3.keys = _Instruction_response_3_allKeys[0].name  # just the first key pressed
            Instruction_response_3.rt = _Instruction_response_3_allKeys[0].rt
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
thisExp.addData('Instruction_response.keys',Instruction_response.keys)
if Instruction_response.keys != None:  # we had a response
    thisExp.addData('Instruction_response.rt', Instruction_response.rt)
thisExp.nextEntry()
# check responses
if Instruction_response_2.keys in ['', [], None]:  # No response was made
    Instruction_response_2.keys = None
thisExp.addData('Instruction_response_2.keys',Instruction_response_2.keys)
if Instruction_response_2.keys != None:  # we had a response
    thisExp.addData('Instruction_response_2.rt', Instruction_response_2.rt)
thisExp.nextEntry()
# check responses
if Instruction_response_3.keys in ['', [], None]:  # No response was made
    Instruction_response_3.keys = None
thisExp.addData('Instruction_response_3.keys',Instruction_response_3.keys)
if Instruction_response_3.keys != None:  # we had a response
    thisExp.addData('Instruction_response_3.rt', Instruction_response_3.rt)
thisExp.nextEntry()
# the Routine "Introduction" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of conditions etc
Training_Blocks = data.TrialHandler(nReps=Training_Repetition, method='sequential', 
    extraInfo=expInfo, originPath=-1,
    trialList=[None],
    seed=None, name='Training_Blocks')
thisExp.addLoop(Training_Blocks)  # add the loop to the experiment
thisTraining_Block = Training_Blocks.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisTraining_Block.rgb)
if thisTraining_Block != None:
    for paramName in thisTraining_Block:
        exec('{} = thisTraining_Block[paramName]'.format(paramName))

for thisTraining_Block in Training_Blocks:
    currentLoop = Training_Blocks
    # abbreviate parameter names if possible (e.g. rgb = thisTraining_Block.rgb)
    if thisTraining_Block != None:
        for paramName in thisTraining_Block:
            exec('{} = thisTraining_Block[paramName]'.format(paramName))
    
    # set up handler to look after randomisation of conditions etc
    Training_Trials = data.TrialHandler(nReps=2.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='Training_Trials')
    thisExp.addLoop(Training_Trials)  # add the loop to the experiment
    thisTraining_Trial = Training_Trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTraining_Trial.rgb)
    if thisTraining_Trial != None:
        for paramName in thisTraining_Trial:
            exec('{} = thisTraining_Trial[paramName]'.format(paramName))
    
    for thisTraining_Trial in Training_Trials:
        currentLoop = Training_Trials
        # abbreviate parameter names if possible (e.g. rgb = thisTraining_Trial.rgb)
        if thisTraining_Trial != None:
            for paramName in thisTraining_Trial:
                exec('{} = thisTraining_Trial[paramName]'.format(paramName))
        
        # --- Prepare to start Routine "Training_Preload_info" ---
        continueRoutine = True
        routineForceEnded = False
        # update component parameters for each repeat
        # Run 'Begin Routine' code from Training_preload
        #Load the randomized images matrix, 
        #each row of the matrix represents a sequence of images,
        #each element should contain a image number,
        #The code below loads the corresponding row of trial from the matrix,
        #please change the path below
        training_trial += 1
        if training_trial > 2:
            training_trial = 1
        training_matrix_name = "sub_" + str(random.randint(0,number_of_sub)) + "_randomized_matrix_702.csv"
        cur_sequence = pd.read_csv(os.path.join(files_path, "Subject_matrix", training_matrix_name), header = None, skiprows = list(range(0,training_trial)) + list(range(training_trial+1,max_trial)) ).iloc[0]
        #cur_sequence_images = cur_sequence.iloc[0]   #Use the indexs to get the images' name
        
        #Load the csv file which contain the correct answers for each trial, 
        #each row contains one answer for a trial, the value should be either 'y' or 'n',
        #The code below loads the corresponding row of trial from the file,
        #please change the path below
        cur_correct = pd.read_csv(os.path.join(files_path, "Correct_answers_702.csv"), header = None, skiprows = list(range(0,training_trial)) + list(range(training_trial+1,max_trial)) ).iloc[0][0]
        #cur_correct = cur_correct_list[0][0]     #Get the correct answer
        
        
        # keep track of which components have finished
        Training_Preload_infoComponents = [Training_Long_gray_screen, Training_Long_Cross]
        for thisComponent in Training_Preload_infoComponents:
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
        
        # --- Run Routine "Training_Preload_info" ---
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *Training_Long_gray_screen* updates
            if Training_Long_gray_screen.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Training_Long_gray_screen.frameNStart = frameN  # exact frame index
                Training_Long_gray_screen.tStart = t  # local t and not account for scr refresh
                Training_Long_gray_screen.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Training_Long_gray_screen, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Training_Long_gray_screen.started')
                Training_Long_gray_screen.setAutoDraw(True)
            if Training_Long_gray_screen.status == STARTED:
                # is it time to stop? (based on local clock)
                if tThisFlip > 2-frameTolerance:
                    # keep track of stop time/frame for later
                    Training_Long_gray_screen.tStop = t  # not accounting for scr refresh
                    Training_Long_gray_screen.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Training_Long_gray_screen.stopped')
                    Training_Long_gray_screen.setAutoDraw(False)
            
            # *Training_Long_Cross* updates
            if Training_Long_Cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Training_Long_Cross.frameNStart = frameN  # exact frame index
                Training_Long_Cross.tStart = t  # local t and not account for scr refresh
                Training_Long_Cross.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Training_Long_Cross, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Training_Long_Cross.started')
                Training_Long_Cross.setAutoDraw(True)
            if Training_Long_Cross.status == STARTED:
                if bool(Training_Long_gray_screen.status == FINISHED):
                    # keep track of stop time/frame for later
                    Training_Long_Cross.tStop = t  # not accounting for scr refresh
                    Training_Long_Cross.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Training_Long_Cross.stopped')
                    Training_Long_Cross.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Training_Preload_infoComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Training_Preload_info" ---
        for thisComponent in Training_Preload_infoComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # Run 'End Routine' code from Trainging_Preload_Images
        Image_loaded_list = []
        for i in range(20):
            
            new_image = visual.ImageStim(
            win=win,
            name='Training_Images'+ str(i), units='pix', 
            image=None, mask=None, anchor='center',
            ori=0.0, pos=(0, 0), size=[image_w, image_h],
            color=[1,1,1], colorSpace='rgb', opacity=None,
            flipHoriz=True, flipVert=False,
            texRes=128.0, interpolate=True, depth=-1.0)
            
            Image_loaded_list.append(new_image)
            del new_image
            Image_loaded_list[i].setImage(cur_sequence[i])
        
        Image_index = 0
        del Training_Images
        Training_Images = Image_loaded_list[Image_index]
        
        #cur_trial_data = {'current_images_list' : cur_sequence_images, 'blank_len' : [len_blank_long] + [len_blank_short] * (images_per_trial-1) }     
        #cur_sequence_images = pd.DataFrame(cur_trial_data)      #Convert the sequence into DataFrame
        #
        ##Save the current images list into the loop file
        #cur_sequence_images.to_csv(os.path.join(files_path, "loopTemplate.csv"),index=False)
        # the Routine "Training_Preload_info" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        Training_Sequences = data.TrialHandler(nReps=20.0, method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=[None],
            seed=None, name='Training_Sequences')
        thisExp.addLoop(Training_Sequences)  # add the loop to the experiment
        thisTraining_Sequence = Training_Sequences.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTraining_Sequence.rgb)
        if thisTraining_Sequence != None:
            for paramName in thisTraining_Sequence:
                exec('{} = thisTraining_Sequence[paramName]'.format(paramName))
        
        for thisTraining_Sequence in Training_Sequences:
            currentLoop = Training_Sequences
            # abbreviate parameter names if possible (e.g. rgb = thisTraining_Sequence.rgb)
            if thisTraining_Sequence != None:
                for paramName in thisTraining_Sequence:
                    exec('{} = thisTraining_Sequence[paramName]'.format(paramName))
            
            # --- Prepare to start Routine "Training" ---
            continueRoutine = True
            routineForceEnded = False
            # update component parameters for each repeat
            # keep track of which components have finished
            TrainingComponents = [Training_Images, Training_GrayScreen, Training_Cross]
            for thisComponent in TrainingComponents:
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
            
            # --- Run Routine "Training" ---
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *Training_Images* updates
                if Training_Images.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    Training_Images.frameNStart = frameN  # exact frame index
                    Training_Images.tStart = t  # local t and not account for scr refresh
                    Training_Images.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(Training_Images, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Training_Images.started')
                    Training_Images.setAutoDraw(True)
                if Training_Images.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > Training_Images.tStartRefresh + 0.1-frameTolerance:
                        # keep track of stop time/frame for later
                        Training_Images.tStop = t  # not accounting for scr refresh
                        Training_Images.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'Training_Images.stopped')
                        Training_Images.setAutoDraw(False)
                
                # *Training_GrayScreen* updates
                if Training_GrayScreen.status == NOT_STARTED and Training_Images.status == FINISHED:
                    # keep track of start time/frame for later
                    Training_GrayScreen.frameNStart = frameN  # exact frame index
                    Training_GrayScreen.tStart = t  # local t and not account for scr refresh
                    Training_GrayScreen.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(Training_GrayScreen, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Training_GrayScreen.started')
                    Training_GrayScreen.setAutoDraw(True)
                if Training_GrayScreen.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > Training_GrayScreen.tStartRefresh + 0.3-frameTolerance:
                        # keep track of stop time/frame for later
                        Training_GrayScreen.tStop = t  # not accounting for scr refresh
                        Training_GrayScreen.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'Training_GrayScreen.stopped')
                        Training_GrayScreen.setAutoDraw(False)
                
                # *Training_Cross* updates
                if Training_Cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    Training_Cross.frameNStart = frameN  # exact frame index
                    Training_Cross.tStart = t  # local t and not account for scr refresh
                    Training_Cross.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(Training_Cross, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Training_Cross.started')
                    Training_Cross.setAutoDraw(True)
                if Training_Cross.status == STARTED:
                    if bool(Training_GrayScreen.status == FINISHED):
                        # keep track of stop time/frame for later
                        Training_Cross.tStop = t  # not accounting for scr refresh
                        Training_Cross.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'Training_Cross.stopped')
                        Training_Cross.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in TrainingComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "Training" ---
            for thisComponent in TrainingComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # Run 'End Routine' code from Update_Training_Images
            Image_index += 1
            
            del TrainingComponents[0]
            del TrainingComponents
            if Image_index < images_per_trial:
            
                del Training_Images
                Training_Images = Image_loaded_list[Image_index]
            
            # the Routine "Training" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
        # completed 20.0 repeats of 'Training_Sequences'
        
        
        # --- Prepare to start Routine "Training_Responses" ---
        continueRoutine = True
        routineForceEnded = False
        # update component parameters for each repeat
        Training_Key_Response.keys = []
        Training_Key_Response.rt = []
        _Training_Key_Response_allKeys = []
        # keep track of which components have finished
        Training_ResponsesComponents = [Training_Responses_Text, Training_Blue_Circle, Training_Green_Circle, Training_Key_Response]
        for thisComponent in Training_ResponsesComponents:
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
        
        # --- Run Routine "Training_Responses" ---
        while continueRoutine and routineTimer.getTime() < 3.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *Training_Responses_Text* updates
            if Training_Responses_Text.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                Training_Responses_Text.frameNStart = frameN  # exact frame index
                Training_Responses_Text.tStart = t  # local t and not account for scr refresh
                Training_Responses_Text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Training_Responses_Text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Training_Responses_Text.started')
                Training_Responses_Text.setAutoDraw(True)
            if Training_Responses_Text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Training_Responses_Text.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    Training_Responses_Text.tStop = t  # not accounting for scr refresh
                    Training_Responses_Text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Training_Responses_Text.stopped')
                    Training_Responses_Text.setAutoDraw(False)
            
            # *Training_Blue_Circle* updates
            if Training_Blue_Circle.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                Training_Blue_Circle.frameNStart = frameN  # exact frame index
                Training_Blue_Circle.tStart = t  # local t and not account for scr refresh
                Training_Blue_Circle.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Training_Blue_Circle, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Training_Blue_Circle.started')
                Training_Blue_Circle.setAutoDraw(True)
            if Training_Blue_Circle.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Training_Blue_Circle.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    Training_Blue_Circle.tStop = t  # not accounting for scr refresh
                    Training_Blue_Circle.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Training_Blue_Circle.stopped')
                    Training_Blue_Circle.setAutoDraw(False)
            
            # *Training_Green_Circle* updates
            if Training_Green_Circle.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                Training_Green_Circle.frameNStart = frameN  # exact frame index
                Training_Green_Circle.tStart = t  # local t and not account for scr refresh
                Training_Green_Circle.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Training_Green_Circle, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Training_Green_Circle.started')
                Training_Green_Circle.setAutoDraw(True)
            if Training_Green_Circle.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Training_Green_Circle.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    Training_Green_Circle.tStop = t  # not accounting for scr refresh
                    Training_Green_Circle.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Training_Green_Circle.stopped')
                    Training_Green_Circle.setAutoDraw(False)
            
            # *Training_Key_Response* updates
            waitOnFlip = False
            if Training_Key_Response.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                Training_Key_Response.frameNStart = frameN  # exact frame index
                Training_Key_Response.tStart = t  # local t and not account for scr refresh
                Training_Key_Response.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Training_Key_Response, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Training_Key_Response.started')
                Training_Key_Response.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(Training_Key_Response.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(Training_Key_Response.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if Training_Key_Response.status == STARTED:
                # is it time to stop? (based on local clock)
                if tThisFlip > 3-frameTolerance:
                    # keep track of stop time/frame for later
                    Training_Key_Response.tStop = t  # not accounting for scr refresh
                    Training_Key_Response.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Training_Key_Response.stopped')
                    Training_Key_Response.status = FINISHED
            if Training_Key_Response.status == STARTED and not waitOnFlip:
                theseKeys = Training_Key_Response.getKeys(keyList=['l', 'k'], waitRelease=False)
                _Training_Key_Response_allKeys.extend(theseKeys)
                if len(_Training_Key_Response_allKeys):
                    Training_Key_Response.keys = _Training_Key_Response_allKeys[0].name  # just the first key pressed
                    Training_Key_Response.rt = _Training_Key_Response_allKeys[0].rt
                    # was this correct?
                    if (Training_Key_Response.keys == str(cur_correct)) or (Training_Key_Response.keys == cur_correct):
                        Training_Key_Response.corr = 1
                    else:
                        Training_Key_Response.corr = 0
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
            for thisComponent in Training_ResponsesComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Training_Responses" ---
        for thisComponent in Training_ResponsesComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # check responses
        if Training_Key_Response.keys in ['', [], None]:  # No response was made
            Training_Key_Response.keys = None
            # was no response the correct answer?!
            if str(cur_correct).lower() == 'none':
               Training_Key_Response.corr = 1;  # correct non-response
            else:
               Training_Key_Response.corr = 0;  # failed to respond (incorrectly)
        # store data for Training_Trials (TrialHandler)
        Training_Trials.addData('Training_Key_Response.keys',Training_Key_Response.keys)
        Training_Trials.addData('Training_Key_Response.corr', Training_Key_Response.corr)
        if Training_Key_Response.keys != None:  # we had a response
            Training_Trials.addData('Training_Key_Response.rt', Training_Key_Response.rt)
        # Run 'End Routine' code from Training_Calculate_Corr
        trial_correct = "WRONG!"
        if Training_Key_Response.corr:
            trial_correct = "CORRECT!"
        else:
            trial_correct = "WRONG!"
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-3.000000)
        
        # --- Prepare to start Routine "Training_Feedback" ---
        continueRoutine = True
        routineForceEnded = False
        # update component parameters for each repeat
        Training_Trial_Feedback.setText(trial_correct)
        # keep track of which components have finished
        Training_FeedbackComponents = [Training_Trial_Feedback]
        for thisComponent in Training_FeedbackComponents:
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
        
        # --- Run Routine "Training_Feedback" ---
        while continueRoutine and routineTimer.getTime() < 0.4:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *Training_Trial_Feedback* updates
            if Training_Trial_Feedback.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Training_Trial_Feedback.frameNStart = frameN  # exact frame index
                Training_Trial_Feedback.tStart = t  # local t and not account for scr refresh
                Training_Trial_Feedback.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Training_Trial_Feedback, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Training_Trial_Feedback.started')
                Training_Trial_Feedback.setAutoDraw(True)
            if Training_Trial_Feedback.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Training_Trial_Feedback.tStartRefresh + 0.4-frameTolerance:
                    # keep track of stop time/frame for later
                    Training_Trial_Feedback.tStop = t  # not accounting for scr refresh
                    Training_Trial_Feedback.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Training_Trial_Feedback.stopped')
                    Training_Trial_Feedback.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Training_FeedbackComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Training_Feedback" ---
        for thisComponent in Training_FeedbackComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # Run 'End Routine' code from Free_Memory_Training
        while len(Image_loaded_list) > 0:
            del Image_loaded_list[0]
        
        for i in range(images_per_trial):
            del cur_sequence[i]
            
        del training_matrix_name
        del cur_sequence
        del cur_correct
        del Image_loaded_list
        del Image_index
        
        gc.collect()
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.400000)
        thisExp.nextEntry()
        
    # completed 2.0 repeats of 'Training_Trials'
    
    
    # --- Prepare to start Routine "End_of_Training_Start_Formal_Experiment" ---
    continueRoutine = True
    routineForceEnded = False
    # update component parameters for each repeat
    Training_End.keys = []
    Training_End.rt = []
    _Training_End_allKeys = []
    # keep track of which components have finished
    End_of_Training_Start_Formal_ExperimentComponents = [end_of_training, Training_End]
    for thisComponent in End_of_Training_Start_Formal_ExperimentComponents:
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
    
    # --- Run Routine "End_of_Training_Start_Formal_Experiment" ---
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *end_of_training* updates
        if end_of_training.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            end_of_training.frameNStart = frameN  # exact frame index
            end_of_training.tStart = t  # local t and not account for scr refresh
            end_of_training.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(end_of_training, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'end_of_training.started')
            end_of_training.setAutoDraw(True)
        
        # *Training_End* updates
        waitOnFlip = False
        if Training_End.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Training_End.frameNStart = frameN  # exact frame index
            Training_End.tStart = t  # local t and not account for scr refresh
            Training_End.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Training_End, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Training_End.started')
            Training_End.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(Training_End.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(Training_End.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if Training_End.status == STARTED and not waitOnFlip:
            theseKeys = Training_End.getKeys(keyList=['space', 'r'], waitRelease=False)
            _Training_End_allKeys.extend(theseKeys)
            if len(_Training_End_allKeys):
                Training_End.keys = _Training_End_allKeys[0].name  # just the first key pressed
                Training_End.rt = _Training_End_allKeys[0].rt
                # was this correct?
                if (Training_End.keys == str('space')) or (Training_End.keys == 'space'):
                    Training_End.corr = 1
                else:
                    Training_End.corr = 0
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
        for thisComponent in End_of_Training_Start_Formal_ExperimentComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "End_of_Training_Start_Formal_Experiment" ---
    for thisComponent in End_of_Training_Start_Formal_ExperimentComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # check responses
    if Training_End.keys in ['', [], None]:  # No response was made
        Training_End.keys = None
        # was no response the correct answer?!
        if str('space').lower() == 'none':
           Training_End.corr = 1;  # correct non-response
        else:
           Training_End.corr = 0;  # failed to respond (incorrectly)
    # store data for Training_Blocks (TrialHandler)
    Training_Blocks.addData('Training_End.keys',Training_End.keys)
    Training_Blocks.addData('Training_End.corr', Training_End.corr)
    if Training_End.keys != None:  # we had a response
        Training_Blocks.addData('Training_End.rt', Training_End.rt)
    # Run 'End Routine' code from Training_End_2
    eeg_trial_num = 0
    if Training_End.corr:
        Training_Blocks.finished = True
    # the Routine "End_of_Training_Start_Formal_Experiment" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
# completed Training_Repetition repeats of 'Training_Blocks'


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
        cur_sequence = pd.read_csv(os.path.join(files_path, "Subject_matrix", matrix_name), header = None, skiprows = list(range(0,eeg_trial_num)) + list(range(eeg_trial_num+1,max_trial)) ).iloc[0]
        #cur_sequence_images = cur_sequence.iloc[0]   #Use the indexs to get the images' name
        #cur_sequence_images = [ os.path.join(images_path, i) for i in cur_sequence_images]
        
        #Load the csv file which contain the correct answers for each trial, 
        #each row contains one answer for a trial, the value should be either 'y' or 'n',
        #The code below loads the corresponding row of trial from the file,
        #please change the path below
        cur_correct = pd.read_csv(os.path.join(files_path, "Correct_answers_702.csv"), header = None, skiprows = list(range(0,eeg_trial_num)) + list(range(eeg_trial_num+1,max_trial)) ).iloc[0][0]
        #cur_correct = cur_correct_list[0][0]     #Get the correct answer
        #eeg_trial_num += 1      #Increase the trial index
        
        gray_trigger_index = 251
        
        print("#Block: ", int(eeg_trial_num/trials_per_block)+1, "\t#Trial: ", int(eeg_trial_num % trials_per_block) ,"\tTotal trials: ", eeg_trial_num+1, "/", max_trial)
        sys.stdout.flush()
        eeg_trial_num += 1
        # keep track of which components have finished
        Load_current_trial_infoComponents = [Long_gray_screen, First_p_port_2, cross_2]
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
            
            # *Long_gray_screen* updates
            if Long_gray_screen.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Long_gray_screen.frameNStart = frameN  # exact frame index
                Long_gray_screen.tStart = t  # local t and not account for scr refresh
                Long_gray_screen.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Long_gray_screen, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Long_gray_screen.started')
                Long_gray_screen.setAutoDraw(True)
            if Long_gray_screen.status == STARTED:
                # is it time to stop? (based on local clock)
                if tThisFlip > 2-frameTolerance:
                    # keep track of stop time/frame for later
                    Long_gray_screen.tStop = t  # not accounting for scr refresh
                    Long_gray_screen.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Long_gray_screen.stopped')
                    Long_gray_screen.setAutoDraw(False)
            # *First_p_port_2* updates
            if First_p_port_2.status == NOT_STARTED and Long_gray_screen.status == STARTED:
                # keep track of start time/frame for later
                First_p_port_2.frameNStart = frameN  # exact frame index
                First_p_port_2.tStart = t  # local t and not account for scr refresh
                First_p_port_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(First_p_port_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'First_p_port_2.started')
                First_p_port_2.status = STARTED
                win.callOnFlip(First_p_port_2.setData, int(gray_trigger_index))
            if First_p_port_2.status == STARTED:
                if frameN >= (First_p_port_2.frameNStart + 1):
                    # keep track of stop time/frame for later
                    First_p_port_2.tStop = t  # not accounting for scr refresh
                    First_p_port_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'First_p_port_2.stopped')
                    First_p_port_2.status = FINISHED
                    win.callOnFlip(First_p_port_2.setData, int())
            
            # *cross_2* updates
            if cross_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                cross_2.frameNStart = frameN  # exact frame index
                cross_2.tStart = t  # local t and not account for scr refresh
                cross_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cross_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cross_2.started')
                cross_2.setAutoDraw(True)
            if cross_2.status == STARTED:
                if bool(Long_gray_screen.status == FINISHED):
                    # keep track of stop time/frame for later
                    cross_2.tStop = t  # not accounting for scr refresh
                    cross_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cross_2.stopped')
                    cross_2.setAutoDraw(False)
            
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
        if First_p_port_2.status == STARTED:
            win.callOnFlip(First_p_port_2.setData, int())
        # Run 'End Routine' code from Preload_Images
        Image_loaded_list = []
        for i in range(20):
            
            new_image = visual.ImageStim(
            win=win,
            name='Images'+ str(i), units='pix', 
            image=None, mask=None, anchor='center',
            ori=0.0, pos=(0, 0), size=[image_w, image_h],
            color=[1,1,1], colorSpace='rgb', opacity=None,
            flipHoriz=True, flipVert=False,
            texRes=128.0, interpolate=True, depth=-1.0)
                
            Image_loaded_list.append(new_image)
            del new_image
            Image_loaded_list[i].setImage(cur_sequence[i])
        
        Image_index = 0
        del Images
        Images = Image_loaded_list[Image_index]
        
        #cur_trial_data = {'current_images_list' : cur_sequence_images, 'blank_len' : [len_blank_long] + [len_blank_short] * (images_per_trial-1) }     
        #cur_sequence_images = pd.DataFrame(cur_trial_data)      #Convert the sequence into DataFrame
        #
        ##Save the current images list into the loop file
        #cur_sequence_images.to_csv(os.path.join(files_path, "loopTemplate.csv"),index=False)
        gray_trigger_index = 252
        
        # the Routine "Load_current_trial_info" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        Sequences = data.TrialHandler(nReps=images_per_trial, method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=[None],
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
            Preload_and_Blank_ScreenComponents = [Images, Image_trigger, blank_screen, gray_trigger, cross]
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
                
                # *Images* updates
                if Images.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
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
                
                # *blank_screen* updates
                if blank_screen.status == NOT_STARTED and Images.status == FINISHED:
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
                    if tThisFlipGlobal > blank_screen.tStartRefresh + 0.3-frameTolerance:
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
                    if bool(blank_screen.status == FINISHED):
                        # keep track of stop time/frame for later
                        cross.tStop = t  # not accounting for scr refresh
                        cross.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'cross.stopped')
                        cross.setAutoDraw(False)
                
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
            if Image_trigger.status == STARTED:
                win.callOnFlip(Image_trigger.setData, int())
            if gray_trigger.status == STARTED:
                win.callOnFlip(gray_trigger.setData, int())
            # Run 'End Routine' code from trigger_update
            #gray_trigger_index = 252
            if cur_triger >= 251:
                cur_triger = 0
            else:
                cur_triger += 1
            
            Image_index += 1
            
            del Preload_and_Blank_ScreenComponents[0]
            del Preload_and_Blank_ScreenComponents
            if Image_index < images_per_trial:
                del Images
                Images = Image_loaded_list[Image_index]
                
            #print(Image_index)
            # the Routine "Preload_and_Blank_Screen" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
        # completed images_per_trial repeats of 'Sequences'
        
        
        # --- Prepare to start Routine "Response" ---
        continueRoutine = True
        routineForceEnded = False
        # update component parameters for each repeat
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
        # keep track of which components have finished
        ResponseComponents = [response_text, blue_circle, Green_circle, key_resp]
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
        while continueRoutine and routineTimer.getTime() < 3.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *response_text* updates
            if response_text.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                response_text.frameNStart = frameN  # exact frame index
                response_text.tStart = t  # local t and not account for scr refresh
                response_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(response_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'response_text.started')
                response_text.setAutoDraw(True)
            if response_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > response_text.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    response_text.tStop = t  # not accounting for scr refresh
                    response_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'response_text.stopped')
                    response_text.setAutoDraw(False)
            
            # *blue_circle* updates
            if blue_circle.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                blue_circle.frameNStart = frameN  # exact frame index
                blue_circle.tStart = t  # local t and not account for scr refresh
                blue_circle.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(blue_circle, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'blue_circle.started')
                blue_circle.setAutoDraw(True)
            if blue_circle.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > blue_circle.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    blue_circle.tStop = t  # not accounting for scr refresh
                    blue_circle.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'blue_circle.stopped')
                    blue_circle.setAutoDraw(False)
            
            # *Green_circle* updates
            if Green_circle.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                Green_circle.frameNStart = frameN  # exact frame index
                Green_circle.tStart = t  # local t and not account for scr refresh
                Green_circle.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Green_circle, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Green_circle.started')
                Green_circle.setAutoDraw(True)
            if Green_circle.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Green_circle.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    Green_circle.tStop = t  # not accounting for scr refresh
                    Green_circle.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Green_circle.stopped')
                    Green_circle.setAutoDraw(False)
            
            # *key_resp* updates
            waitOnFlip = False
            if key_resp.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
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
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-3.000000)
        
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
        # Run 'End Routine' code from Free_Memeory
        while len(Image_loaded_list) > 0:
            del Image_loaded_list[0]
        
        for i in range(images_per_trial):
            del cur_sequence[i]
        
        del cur_sequence
        del cur_correct
        del Image_loaded_list
        del Image_index
        gc.collect()
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.400000)
        thisExp.nextEntry()
        
    # completed trials_per_block repeats of 'Trials'
    
    
    # --- Prepare to start Routine "Block_Feedback" ---
    continueRoutine = True
    routineForceEnded = False
    # update component parameters for each repeat
    # Run 'Begin Routine' code from accuracy_rate
    num_correct = round( (num_correct / trials_per_block) * 100 , 2)       #Calculate accuracy rate
    accuracy_rate = str("Your accuracy rate is ") + str(num_correct) + str("%")
    text_3.setText(accuracy_rate)
    # keep track of which components have finished
    Block_FeedbackComponents = [text_3]
    for thisComponent in Block_FeedbackComponents:
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
    
    # --- Run Routine "Block_Feedback" ---
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
        for thisComponent in Block_FeedbackComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Block_Feedback" ---
    for thisComponent in Block_FeedbackComponents:
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
    block_end_text = "This is the end of block " + str(cur_block)
    
    num_correct = 0     #initialize the number of correct answers
    
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
            if tThisFlipGlobal > Rest.tStartRefresh + 45-frameTolerance:
                # keep track of stop time/frame for later
                Rest.tStop = t  # not accounting for scr refresh
                Rest.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Rest.stopped')
                Rest.setAutoDraw(False)
        
        # *PressKey_to_Continue* updates
        if PressKey_to_Continue.status == NOT_STARTED and tThisFlip >= 48-frameTolerance:
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
        if key_resp_2.status == NOT_STARTED and tThisFlip >= 48-frameTolerance:
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
            theseKeys = key_resp_2.getKeys(keyList=['space'], waitRelease=False)
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


# --- Prepare to start Routine "End_of_experiment" ---
continueRoutine = True
routineForceEnded = False
# update component parameters for each repeat
# keep track of which components have finished
End_of_experimentComponents = [etRecord_end, text]
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
    
    # *text* updates
    if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        text.frameNStart = frameN  # exact frame index
        text.tStart = t  # local t and not account for scr refresh
        text.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'text.started')
        text.setAutoDraw(True)
    if text.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > text.tStartRefresh + 3-frameTolerance:
            # keep track of stop time/frame for later
            text.tStop = t  # not accounting for scr refresh
            text.frameNStop = frameN  # exact frame index
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text.stopped')
            text.setAutoDraw(False)
    
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
