#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2022.2.5),
    on January 30, 2023, at 11:42
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
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
    originPath='C:\\Users\\15202\\OneDrive\\C_\\University of Amsterdam\\Intern\\EEG_TargetDetection_Experiment.py',
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
    size=[1368, 912], fullscr=True, screen=0, 
    winType='pyglet', allowStencil=False,
    monitor='Surface', color=[0,0,0], colorSpace='rgb',
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

# Setup iohub keyboard
ioConfig['Keyboard'] = dict(use_keymap='psychopy')

ioSession = '1'
if 'session' in expInfo:
    ioSession = str(expInfo['session'])
ioServer = io.launchHubServer(window=win, **ioConfig)
eyetracker = None

# create a default keyboard (e.g. to check for escape)
defaultKeyboard = keyboard.Keyboard(backend='iohub')

# --- Initialize components for Routine "Initialization" ---

# --- Initialize components for Routine "Introduction" ---
text_2 = visual.TextStim(win=win, name='text_2',
    text='Intro',
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);

# --- Initialize components for Routine "Correctness_Initialization" ---

# --- Initialize components for Routine "Preload" ---

# --- Initialize components for Routine "Blank" ---
blank = visual.Rect(
    win=win, name='blank',units='pix', 
    width=[(win.size)[0], (win.size)[1]][0], height=[(win.size)[0], (win.size)[1]][1],
    ori=0.0, pos=(0, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor=[0.0039, 0.0039, 0.0039], fillColor=[0.0039, 0.0039, 0.0039],
    opacity=None, depth=0.0, interpolate=True)
polygon = visual.ShapeStim(
    win=win, name='polygon', vertices='cross',
    size=(0.01, 0.01),
    ori=0.0, pos=(0, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='red', fillColor='red',
    opacity=None, depth=-1.0, interpolate=True)

# --- Initialize components for Routine "Sequence" ---
Images = visual.ImageStim(
    win=win,
    name='Images', units='pix', 
    image='sin', mask=None, anchor='center',
    ori=0.0, pos=(0, 0), size=[(win.size)[0], (win.size)[1]],
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=True, flipVert=False,
    texRes=128.0, interpolate=True, depth=0.0)
Cross = visual.ShapeStim(
    win=win, name='Cross', vertices='cross',
    size=(0.01, 0.01),
    ori=0.0, pos=(0, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor=[1.0000, -1.0000, -1.0000], fillColor=[1.0000, -1.0000, -1.0000],
    opacity=1.0, depth=-1.0, interpolate=True)
Rectangle = visual.Rect(
    win=win, name='Rectangle',units='pix', 
    width=[(win.size)[0], (win.size)[1]][0], height=[(win.size)[0], (win.size)[1]][1],
    ori=0.0, pos=(0, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor=[0.0039, 0.0039, 0.0039], fillColor=[0.0039, 0.0039, 0.0039],
    opacity=0.0, depth=-2.0, interpolate=True)

# --- Initialize components for Routine "Response" ---
text = visual.TextStim(win=win, name='text',
    text='Did the target presents?\n\nYes? Press y\n\nNo? Press n',
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
key_resp = keyboard.Keyboard()

# --- Initialize components for Routine "Feedback" ---
text_3 = visual.TextStim(win=win, name='text_3',
    text='',
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-1.0);

# --- Initialize components for Routine "End" ---
text_4 = visual.TextStim(win=win, name='text_4',
    text='End of one block',
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine 

# --- Prepare to start Routine "Initialization" ---
continueRoutine = True
routineForceEnded = False
# update component parameters for each repeat
# Run 'Begin Routine' code from Initialization_code
import pandas as pd
import os

eeg_trial_num = 0   #Track with the current trial
images_per_trial = 3
trials_per_block = 5
blocks_per_repetition = 1
total_repetition = 1
total_blocks = blocks_per_repetition * total_repetition
max_trial = trials_per_block * blocks_per_repetition * total_repetition #The maximum of trials

#Get the list of all images' file name
#Please change the path below
Images_name_list = os.listdir(r"C:\Users\15202\OneDrive\C_\University of Amsterdam\Intern\Pictures")

# NOTICE: The Images from the list are not in numerical order, 
# i.e. the third element is "10.tiff" but not "2.tiff"
# In the testing case (and also formal experiment), this would not cause any problem. 
# However, if we need to use these indices to track with specific images, 
# use the command below to sort the list
# (make sure the prefix of file name is number only).

#sorted(Images_name_list, key = lambda item: int(item.split(".")[0]))
# keep track of which components have finished
InitializationComponents = []
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
    # keep track of which components have finished
    IntroductionComponents = [text_2]
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
    while continueRoutine and routineTimer.getTime() < 2.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_2* updates
        if text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_2.frameNStart = frameN  # exact frame index
            text_2.tStart = t  # local t and not account for scr refresh
            text_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_2.started')
            text_2.setAutoDraw(True)
        if text_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_2.tStartRefresh + 2-frameTolerance:
                # keep track of stop time/frame for later
                text_2.tStop = t  # not accounting for scr refresh
                text_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_2.stopped')
                text_2.setAutoDraw(False)
        
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
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-2.000000)
    
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
            
            # --- Prepare to start Routine "Preload" ---
            continueRoutine = True
            routineForceEnded = False
            # update component parameters for each repeat
            # Run 'Begin Routine' code from Get_Images_list
            #Load the randomized images matrix, 
            #each row of the matrix represents a sequence of images,
            #each element should contain a image number,
            #The code below loads the corresponding row of trial from the matrix,
            #please change the path below
            cur_sequence = pd.read_csv(r'C:\Users\15202\OneDrive\C_\University of Amsterdam\Intern\Randomized matrix.csv', header = None, skiprows = list(range(0,eeg_trial_num)) + list(range(eeg_trial_num+1,max_trial)) )
            cur_sequence_images = [Images_name_list[cur_sequence.iloc[0, i]] for i in range(images_per_trial) ]     #Use the indexs to get the images' name
            cur_sequence_images = [ "Pictures\\" + i for i in cur_sequence_images]      #I stoerd all the images in the Picture folder, the path can be changed according to your case
            
            #Load the csv file which contain the correct answers for each trial, 
            #each row contains one answer for a trial, the value should be either 'y' or 'n',
            #The code below loads the corresponding row of trial from the file,
            #please change the path below
            cur_correct = pd.read_csv(r'C:\Users\15202\OneDrive\C_\University of Amsterdam\Intern\Correct_answers.csv', header = None, skiprows = list(range(0,eeg_trial_num)) + list(range(eeg_trial_num+1,max_trial)) )
            cur_correct = cur_correct[0][0]     #Get the correct answer
            eeg_trial_num += 1      #Increase the trial index
            
            cur_trial_data = {'current_images_list' : cur_sequence_images}     
            cur_sequence_images = pd.DataFrame(cur_trial_data)      #Convert the sequence into DataFrame
            
            #Save the current images list into the loop file
            cur_sequence_images.to_csv(r'C:\Users\15202\OneDrive\C_\University of Amsterdam\Intern\loopTemplate.csv',index=False)
            
            
            
            # keep track of which components have finished
            PreloadComponents = []
            for thisComponent in PreloadComponents:
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
            
            # --- Run Routine "Preload" ---
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
                for thisComponent in PreloadComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "Preload" ---
            for thisComponent in PreloadComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # the Routine "Preload" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "Blank" ---
            continueRoutine = True
            routineForceEnded = False
            # update component parameters for each repeat
            # keep track of which components have finished
            BlankComponents = [blank, polygon]
            for thisComponent in BlankComponents:
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
            
            # --- Run Routine "Blank" ---
            while continueRoutine and routineTimer.getTime() < 2.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *blank* updates
                if blank.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    blank.frameNStart = frameN  # exact frame index
                    blank.tStart = t  # local t and not account for scr refresh
                    blank.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(blank, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'blank.started')
                    blank.setAutoDraw(True)
                if blank.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > blank.tStartRefresh + 2-frameTolerance:
                        # keep track of stop time/frame for later
                        blank.tStop = t  # not accounting for scr refresh
                        blank.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'blank.stopped')
                        blank.setAutoDraw(False)
                
                # *polygon* updates
                if polygon.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    polygon.frameNStart = frameN  # exact frame index
                    polygon.tStart = t  # local t and not account for scr refresh
                    polygon.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(polygon, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'polygon.started')
                    polygon.setAutoDraw(True)
                if polygon.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > polygon.tStartRefresh + 2-frameTolerance:
                        # keep track of stop time/frame for later
                        polygon.tStop = t  # not accounting for scr refresh
                        polygon.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'polygon.stopped')
                        polygon.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in BlankComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "Blank" ---
            for thisComponent in BlankComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-2.000000)
            
            # set up handler to look after randomisation of conditions etc
            Sequences = data.TrialHandler(nReps=1.0, method='random', 
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
                
                # --- Prepare to start Routine "Sequence" ---
                continueRoutine = True
                routineForceEnded = False
                # update component parameters for each repeat
                Images.setImage(current_images_list)
                # keep track of which components have finished
                SequenceComponents = [Images, Cross, Rectangle]
                for thisComponent in SequenceComponents:
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
                
                # --- Run Routine "Sequence" ---
                while continueRoutine and routineTimer.getTime() < 1.5:
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
                        if tThisFlipGlobal > Images.tStartRefresh + 0.5-frameTolerance:
                            # keep track of stop time/frame for later
                            Images.tStop = t  # not accounting for scr refresh
                            Images.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'Images.stopped')
                            Images.setAutoDraw(False)
                    
                    # *Cross* updates
                    if Cross.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                        # keep track of start time/frame for later
                        Cross.frameNStart = frameN  # exact frame index
                        Cross.tStart = t  # local t and not account for scr refresh
                        Cross.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(Cross, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'Cross.started')
                        Cross.setAutoDraw(True)
                    if Cross.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > Cross.tStartRefresh + 1.5-frameTolerance:
                            # keep track of stop time/frame for later
                            Cross.tStop = t  # not accounting for scr refresh
                            Cross.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'Cross.stopped')
                            Cross.setAutoDraw(False)
                    
                    # *Rectangle* updates
                    if Rectangle.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                        # keep track of start time/frame for later
                        Rectangle.frameNStart = frameN  # exact frame index
                        Rectangle.tStart = t  # local t and not account for scr refresh
                        Rectangle.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(Rectangle, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'Rectangle.started')
                        Rectangle.setAutoDraw(True)
                    if Rectangle.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > Rectangle.tStartRefresh + 1-frameTolerance:
                            # keep track of stop time/frame for later
                            Rectangle.tStop = t  # not accounting for scr refresh
                            Rectangle.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'Rectangle.stopped')
                            Rectangle.setAutoDraw(False)
                    
                    # check for quit (typically the Esc key)
                    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                        core.quit()
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in SequenceComponents:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "Sequence" ---
                for thisComponent in SequenceComponents:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                if routineForceEnded:
                    routineTimer.reset()
                else:
                    routineTimer.addTime(-1.500000)
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
            ResponseComponents = [text, key_resp]
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
                
                # *key_resp* updates
                waitOnFlip = False
                if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
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
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > key_resp.tStartRefresh + 3-frameTolerance:
                        # keep track of stop time/frame for later
                        key_resp.tStop = t  # not accounting for scr refresh
                        key_resp.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'key_resp.stopped')
                        key_resp.status = FINISHED
                if key_resp.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp.getKeys(keyList=['y' ,'n'], waitRelease=False)
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
            if key_resp.corr:
                num_correct += 1        #Record how many times the participants reponse correctly
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-3.000000)
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
        
        # --- Prepare to start Routine "End" ---
        continueRoutine = True
        routineForceEnded = False
        # update component parameters for each repeat
        # keep track of which components have finished
        EndComponents = [text_4]
        for thisComponent in EndComponents:
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
        
        # --- Run Routine "End" ---
        while continueRoutine and routineTimer.getTime() < 3.0:
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
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in EndComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "End" ---
        for thisComponent in EndComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-3.000000)
        thisExp.nextEntry()
        
    # completed blocks_per_repetition repeats of 'Blocks'
    
    thisExp.nextEntry()
    
# completed total_repetition repeats of 'Repetition'


# --- End experiment ---
# Flip one final time so any remaining win.callOnFlip() 
# and win.timeOnFlip() tasks get executed before quitting
win.flip()

# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename+'.csv', delim='auto')
thisExp.saveAsPickle(filename)
logging.flush()
# make sure everything is closed down
if eyetracker:
    eyetracker.setConnectionState(False)
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()
