"""

Author: Griffin Dugan
Brandeis Memory & Cognition (Wingfield) Lab
EVE1 Main Program


Description: This is the main program for the EVE1 experiment. It is a GUI program that allows for the running of the experiment, as well as the analysis of the data. It is built in Python using the tkinter library for the GUI, and uses OpenCV for video processing, as well as PyDub for audio processing. It also uses the PIL library for image processing and viewing.



TODO LIST:

- TODO fix weird time after in experiment FIXED i think?
- TODO fix back in flags DONE
- TODO fix first flag error
- TODO update for mulitple passages at once to work with new array and flags
- TODO check for correct number of video files uploaded for set DONE

- TODO: it might be worthwhile to write a better fix flags check for flags within a range of eachother.

"""

# Import Statements
import tkinter as tk # importing tkinter as a whole for GUI 
from tkinter import ttk, filedialog # importing specific tk modules
import cv2 # importing open-cv for video recording and processing (as well as image manipulation)
            # ! This could potentially be why there's a memory leak tho.
from PIL import ImageTk, Image # importing pillow for better image work as well as image viewing in TK gui 
import csv # importing csv for saving file
import os # importing os for file work etc.
from pydub import AudioSegment # importing pydub "proper", audiosegment only, for getting audio loaded
from pydub.playback import play # importing pydub playback, play only, for playing audio
import threading # importing threading for multithreading during testing
                # * As well as planned multi-passage analysis
import queue
import math # importing math cuz math
import statistics as stat

import numpy as np
from collections import defaultdict as dd
import time
import imageio.v3 as iio
import pandas as pd
import matplotlib.pyplot as plt
import tables

from OCR import findText, timeIt # These are all functions pulled from the OCR.py file in the same repository.
from parsing import threadedParse


# generic global variables needed
experiment = True # determines whether analysing or testing

LARGEFONT =("Verdana", 35) # generic heading font
PDiffs = {1:"0", 2:"0", 3:"0", 4:"0", 5:"0", 6:"0", 7:"0", 8:"0", 0:"P",  # These are the difficulty ratings of each of the passages.
          9:"1", 10:"1", 11:"1", 12:"1", 13:"1", 14:"1", 15:"1", 16:"1",
          17:"1", 18:"1", 19:"1", 20:"1", 21:"1", 22:"1", 23:"1", 24:"1"}

# Initisalising storing matrixes
TbfPrecursor, TfPrecursor, TrPrecursor, TffPrecursor = np.empty((17),dtype=np.ndarray),np.empty((17),dtype=np.ndarray),np.empty((17),dtype=np.ndarray),np.empty((17),dtype=np.ndarray)
bfPrecursor, fPrecursor, rPrecursor, ffPrecursor = np.empty((17),dtype=np.ndarray),np.empty((17),dtype=np.ndarray),np.empty((17),dtype=np.ndarray),np.empty((17),dtype=np.ndarray)
F, R, FF, bF = None, None, None, None
tF, tR, tFF, tbF = None, None, None, None

"""These are the storing silos for each of the necessary data points throughout analysis. (bF, F, R)

## bF
:var bF: This is the batched flags dictionary. It holds key as passage and then an item as another dictionary. 
        This other dictionary is the dictionary of batched flags for the particular passage. Batched flags are flags that are the same value that are all right next to eachother, which generally implies that they are all the same result, so we can batch them together for quicker manual fixing (in FlagsPg).
        It holds the key labelled as the first flag of its type in a row with the index in F[passage] directly after it. So, a flag of `FLAG: 4` at index in F[passage] `91` would be `FLAG: 491`.
        This is a little bit messy, but it was so that each flag would guarenteed have a unique key name.
            This is entirely used in 
:type bF: dict[dict]

## F
:var F: This is the flags dictionary. It holds the key as the passage and then an item as a list.
        This list is the list of all flags in the particular passage. Flags are erroneous results.
        Each list index is another list, made up of index in R (int), value of the flag (str), and the actual CV2 formatted image (list[list]).
            This is primarily used in the `FlagsPg` frame, so having the image on hand to present to the user is necessary. 
:type F: dict[list[list]]

## R
:var R: This is the results dictionary. It holds the key as the passage and then an item as a list.
        This list is the list of all the results, or number by frame, in the particular passage.
        This is then edited as flags are dealt with.
:type R: dict[list]
"""
# TODO update the above
Orders = np.array([ # 8 running orders
    (0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(0,3),(1,3)
])

listenOrders = np.array([ # These are the running orders of the listening passages.
    # 4  5  9  15 2  14 12 1  13 8  16 10 3  6  11 7
    # h  h  l  l  h  l  l  h  l  h  l  l  h  h  l  h

    # 14 12 5  8  11 1  2  15 6  13 4  7  16 10 3  9
    # l  l  h  h  l  h  h  l  h  l  h  h  l  l  h  l 
    [0,4,5,9,15,2,14,12,1,13,8,16,10,3,6,11,7],
    [0,14,12,5,8,11,1,2,15,6,13,4,7,16,10,3,9]
],dtype=np.int_)

expOrders = np.array([ # These are the running orders of the experimental passages
    # 17 20 19 18 24 22 21 23
    # 1h 1h 1h 1h 2l 2l 2l 2l

    # 22 24 23 21 19 20 17 18
    # 2l 2l 2l 2l 1h 1h 1h 1h

    # 17 18 19 20 21 22 23 24
    # 1l 1l 1l 1l 2h 2h 2h 2h

    # 24 21 23 22 18 20 17 19
    # 2h 2h 2h 2h 1l 1l 1l 1l
    [17,20,19,18,24,22,21,23],
    [22,24,23,21,19,20,17,18],
    [19,17,18,20,22,24,21,23],
    [24,21,23,22,18,20,17,19]
])


runningOrders = np.array([
    #p y n  n  y  y  n  y  y n  y  n  y  y  n  y n  y  n  y n  y y  n  y n  y  y n  n  y n  y  n  n  n  y  n  n  y n
    [0,9,21,29,15,14,27,18,2,35,13,39,12,10,31,7,30,17,33,1,26,6,19,38,8,24,11,4,25,34,5,28,16,40,22,32,20,23,37,3,36],
    #p n  y  y  n  n  y  n n y  n  y  n  n  y  n y  n  y  n y  n n  y  n y  n  n y  y  n y  n  y  y  y  n  y  y  n y
    [0,9,21,29,15,14,27,18,2,35,13,39,12,10,31,7,30,17,33,1,26,6,19,38,8,24,11,4,25,34,5,28,16,40,22,32,20,23,37,3,36]
])


singlePassage = False # This determines whether a single passage is being analysed or it's all of them at once.
files = [] # This is the storage of all uploaded video files. -- Only used during non-single passage.

base_dir = os.path.dirname(__file__) # This is the base directory to grab files from. (Needed for application work.)


def CVtoPIL(CVimage:list) -> list:
    """This converts a cv2 image into a Pillow image

    :param CVimage: The cv2 image.
    :type CVimage: list
    :return: The PIL image.
    :rtype: list
    """
    color_coverted = cv2.cvtColor(CVimage, cv2.COLOR_BGR2RGB) 
    return Image.fromarray(color_coverted)


def calcMean(res: list) -> list:
    return round(stat.fmean(res)) if len(res) > 0 else 0

def calcMeanIQR(res:list) -> int:
    """Calculates the new mean after removing outliers from a dataset.

    :param res: Sorted list of final results after each test
    :type res: list
    :return: The new mean, with outliers removed.
    :rtype: int
    """
    med = len(res)//2
    q1, q3 = stat.median(res[:med]), stat.median(res[med:])
    upperFence, lowerFence = q3 + (1.5 * (q3 - q1)), q1 - (1.5 * (q3 - q1))
    res = [i for i in res if i <= upperFence and i >= lowerFence]
    return round(stat.fmean(res))

def calcMeanZ(res:list) -> float:
    """i honestly don't know how z score outliers are supposed to work, as I couldn't get a clear outlier removed

    :param res: Sorted list of final results after each test
    :type res: list
    :return: The new mean, with outliers removed.
    :rtype: int
    """
    sd = stat.stdev(res)
    ave = stat.fmean(res)
    z = []
    for x in res:
        z.append((x - ave)/sd)
    res = [x for x in res if zS for zS in z if zS < 3 and zS > -3]
            



# Master GUI Class (controller)
class tkinterApp(tk.Tk):
    """
    This class defines the gui window and framing for every other frame.
    """
    def __init__(self, *args, **kwargs):
        # init function for class Tk
        tk.Tk.__init__(self, *args, **kwargs)

        # this is shared data between the frames
        self.shared_data = {
            "video": tk.StringVar(), # fileName of video
            "mov": tk.BooleanVar(), # whether it is video or not -- this is not necessary any longer -- needs to be removed
            "ID": tk.StringVar(), # the participant ID
            "Passage": tk.IntVar(), # the current passage number
            "PassageOrder": tk.IntVar(), # the passage set order
            "crop": tk.StringVar() # crop values
        }

        # briefly setting base data points that are accessed before they're set by user
        self.shared_data["video"].set("NONE")
        self.shared_data["crop"].set("NONE")

        # This is OLD? prevention of playing multiple audio files at once.
        # ? I think this isn't needed, but I'm not sure.
        self.playingSet = False
        self.playingOne = False
        self.playing = False

        self.readyPlay = tk.BooleanVar()

        # creating a gui window
        container = tk.Frame(self)  # parent throughout windows
        container.pack(side = "top", fill = "both", expand = True) 
        container.grid_rowconfigure(0, weight = 1,uniform="fred") # i am now afraid to mess with any of this after breaking the gui by touching this later, so it is what it is for now.
        container.grid_columnconfigure(0, weight = 1,uniform="fred") # i think "fred" is not a necessary name, but just any uniform name that is kept true throughout all rows.

        # Universal Styles
        self.style = ttk.Style()
        self.style.configure("DisabledB.TButton",foreground = '#999999',background = "#bcbcbc")
        self.style.configure("BlueB.TButton", background = 'blue') # doesn't work on MacOS -- test on windows?
  
        # initialising frames to an empty array
        self.frames = {}  # basically declares that each new window, or frame, is part of this master GUI window framing.

        # Initialise debug first
        frame = Debug(container, self)
        self.frames[Debug] = frame 

        frame.grid(row = 0, column = 0, sticky ="nsew")
        # Centring the rows
        frame.grid_rowconfigure(0, weight=1,uniform="fred")
        frame.grid_rowconfigure(10, weight=1,uniform="fred")
        frame.grid_columnconfigure(0, weight=1,uniform="fred")
        frame.grid_columnconfigure(10, weight=1,uniform="fred")

        # Setting up all of the windows
        # This grabs each frame and correctly frames it via this master GUI.
        for F in (StartPg, PassagePg, UploadPg, WorkingPg, FinalPg, FlagsPg, DownPg, PassageSetPg, PSelectPg, RunExpPg, OnePSelectPg, Run1PassagePg,CamAdjustPg,cropPg,TestComplete,WorkingSetPg): # * if adding new window, must add to this tuple
            frame = F(container, self, self.frames[Debug])
            self.frames[F] = frame 
  
            frame.grid(row = 0, column = 0, sticky ="nsew")
            # Centring the rows
            frame.grid_rowconfigure(0, weight=1,uniform="fred")
            frame.grid_rowconfigure(10, weight=1,uniform="fred")
            frame.grid_columnconfigure(0, weight=1,uniform="fred")
            frame.grid_columnconfigure(10, weight=1,uniform="fred")

        self.title('griffin pilot idk') # title at top of screen
        # ! ought to change this soon, but given lack of name....

        def debug():
            self.show_frame(Debug)

        def crop(): # This is an old way to crop the image 
            # TODO: no longer necessary.
            if self.shared_data["video"].get() == "NONE": print("No")
            else: self.show_frame(cropPg)

        # Adding Settings Menu and commands 
        menubar = tk.Menu(self)
        self.settings = tk.Menu(menubar, tearoff = 0) 
        menubar.add_cascade(label ='Settings', menu = self.settings) 
        self.settings.add_command(label ='Debug', command = debug) 
        self.settings.add_command(label ='Set Image Crop', command = crop) 
        self.settings.entryconfigure("Set Image Crop", state=tk.DISABLED)
        self.settings.add_separator() 
        self.settings.add_command(label ='Exit', command = self.destroy) 

        self.config(menu = menubar) 

        # showing the start page
        self.show_frame(StartPg)


    # to display the current frame passed as parameter
    def show_frame(self, cont):
        """Showing a new window

        :param cont: The next window class/frame
        :type cont: class (but python doesn't like me saying that)
        """
        if cont not in self.frames:
            self.frames[cont] = frame = cont(self.container, self) # * IF THIS RUNS AN ERROR, YOU HAVE NOT ADDED THE NEW FRAME TO THE TUPLE OF FRAMES IN THE MASTER GUI CLASS.
                # if it runs an issue and you have, good luck idk 
                # fyi, i'm supposed to make container in the GUI class above into self.container and then this would work as intended, but currently it's fine.
            frame.grid(row=0, column=0, sticky="nsew")
        frame = self.frames[cont]
        frame.tkraise()



# debug
class Debug(tk.Frame):
     
    def __init__(self, parent, controller):
        self.controller = controller
        tk.Frame.__init__(self, parent)


        self.f = {}
        
        label = ttk.Label(self, text ="This is a new Window", font=LARGEFONT)
        label.grid(row=0,column=3,padx=10,pady=10)

        leftL = ttk.Label(self, text ="Run Experiment")
        leftL.grid(row=1,column=2,padx=10,pady=10)

        rightL = ttk.Label(self, text ="Analyse Data")
        rightL.grid(row=1,column=4,padx=10,pady=10)

        def blueNext():
            button2.config(style="BlueB.TButton",state = 'enabled')

        # Where to go checkboxes 
        v = tk.IntVar(self, 10) 
        # Dictionaries to create multiple buttons -- These are the location options
        valuesL = {"RO Select" : 1, 
                "P Select" : 2,
                "Listening" : 3,
                "Experiment" : 4}
        valuesR = {"P Select" : 5,
                "Final RO" : 6,
                "Final S" : 7,
                "Flags RO" : 8}
        for (text, value) in valuesL.items(): # creating check buttons, left side
            tk.Radiobutton(self, text = text, variable = v, value = value, command=blueNext).grid(row=int(value+2),column=2,padx=10,pady=10)
        for (text, value) in valuesR.items(): # creating check buttons, right side
            tk.Radiobutton(self, text = text, variable = v, value = value, command=blueNext).grid(row=int(value-2),column=4,padx=10,pady=10)

        def nextF():
            """
            On next, save passage number and move to next window (upload)
            """
            def startPage(single, exp):
                global singlePassage, experiment
                controller.shared_data["ID"].set("DEBUG")
                PATH = os.path.join(base_dir, 'Video Files/PDEBUG')
                if not os.path.exists(PATH):
                    os.makedirs(PATH)
                singlePassage = single
                experiment = exp

            def ROselect():
                controller.shared_data["PassageOrder"].set(0)
                


            where = v.get() # stores where to go pick 
            if where == 1:
                startPage(single=False, exp=True)
                controller.show_frame(PSelectPg)
            elif where == 2: 
                startPage(single=True, exp=True)
                controller.show_frame(OnePSelectPg)
            elif where == 3:
                startPage(single=False, exp=True)
                ROselect()
                controller.playingSet = True
                controller.show_frame(PSelectPg)

                passPage = self.f["EsetSel"].page
                passPage.page.resetGrid()
                passPage.playGrid()
                if controller.playingSet and not controller.playing: 
                    passPage.page.playing()
            elif where == 4:
                startPage(single=False, exp=True)
                ROselect()
                controller.playingSet = True
                controller.show_frame(PSelectPg)

                passPage = self.f["EsetSel"].page
                passPage.resetGrid()
                passPage.playGrid()
                passPage.run = 17
                passPage.passageNum = listenOrders[Orders[1][0]][17]
                for i in range(7):
                    passPage.volumes[0, i] = 34
                    passPage.volumes[1, i] = 40
                passPage.volumes[1, 7] = 40
                passPage.vol.set("34")
                passPage.heading.grid(row = 0, column = 5, padx = 10, pady = 10)
                passPage.ready.grid(row = 3, column = 5, padx = 10, pady = 10)
                passPage.finishListen()
            elif where == 5:
                startPage(single=False, exp=False)
                controller.show_frame(PassageSetPg)
            elif where == 6:
                startPage(single=False, exp=False)
                ROselect()

                




            






        # Next Button
        button2 = ttk.Button(self, text ="Next", style="DisabledB.TButton", command=nextF, state = 'disabled')
        button2.grid(row = 10, column = 5, padx = 10, pady = 10)
        
    class frame():
        def __init__(self, page:object, name:str):
            self.page = page
            self.name = name

    def initFrame(self, page, name):
        self.f[name] = self.frame(page,name)


# Actual Windows:

# 1st window frame startpage  
class StartPg(tk.Frame):
    """
    This is simply the start page, where the user selects analysing or testing, as well as single passage or set of passages. They also input participant ID here.
    """
    def __init__(self, parent, controller, debug): 
        self.controller = controller
        tk.Frame.__init__(self, parent)
        self.debug = debug
        
        self.debug.initFrame(page=self,name="Start")

        # Heading
        label = ttk.Label(self, text ="Start Page", font = LARGEFONT)
        label.grid(row = 0, column = 3, padx = 10, pady = 20, sticky="") 

        def submit():
            """
            On submit, save ID and whether actively experimenting or analysing data
            """
            global experiment, singlePassage
            partID = str(ID.get())
            ID.set("")
            if partID != "": # if the ID hasn't been inputted, make sure it does
                self.controller.shared_data["ID"].set(partID)
                # * if need to reset ID, put here. 
                # * I don't think I will need to reset ID, but unsure.
                if experiment: # if running the experiment
                    # If there's no folder for the participant's results, make one.
                    PATH = f'Video Files/P{partID}'
                    PATH = os.path.join(base_dir, PATH)
                    if not os.path.exists(PATH):
                        os.makedirs(PATH)
    
                    # Moving to passage select frame.
                    if singlePassage: # If only running a single passage, move to that frame, else move to running order frame.
                        self.controller.show_frame(OnePSelectPg)
                    else:
                        self.controller.show_frame(PSelectPg)
                else: # Else, analyse data
                    if singlePassage: # if analysing single passage, select passage, else, select running order.
                        self.controller.show_frame(PassagePg)
                    else: 
                        self.controller.show_frame(PassageSetPg)
            else: # if ID wasn't inputted into entry box
                # this shuffles most of the widgets down in order to place the dialog to user that they need an ID
                checkbutton1.grid(row=5, column=3, padx=10, pady=10)
                checkbutton.grid(row=4, column=3, padx=10, pady=10)
                button1.grid(row = 3, column = 3, padx = 10, pady = 10)
                NoID.grid(row = 2, column = 3, padx = 10, pady = 10)
                # ! might need to add `Space` shuffling here, unsure. Depends on formatting.
            
        ## Start button
        button1 = ttk.Button(self, text ="Get Started", command = submit)
        button1.grid(row = 2, column = 3, padx = 10, pady = 10)

        def on_button_toggle():
            """
            This is for the "Analyse Data" checkbox.
            On clicking the toggle for whether actively experimenting, save that info.
            """
            global experiment
            if var.get() == 1: # if checked, turn off experiemnt
                experiment = False
            else: # else keep it on
                experiment = True

        # Setting up the experiment/analyse checkbox
        var = tk.IntVar() # ye ik these are not well named variables, but they're only ever used like once.
        checkbutton = tk.Checkbutton(self, text="Analyse Data", variable=var, onvalue=1, offvalue=0, command=on_button_toggle)
        checkbutton.config(selectcolor="green", relief="raised")
        checkbutton.grid(row=3, column=3, padx=10, pady=10)
        checkbutton.flash()


        
        def on_button_toggle1():
            """
            This is for the "Enable Single Passage Mode" checkbox.
            On clicking the toggle for single passage, save that info.
            """
            global singlePassage
            if var1.get() == 1: # if checkec, turn on single passage
                singlePassage = True
            else: # else keep it off
                singlePassage = False

        # Setting up the single passage checkbox
        var1 = tk.IntVar()
        checkbutton1 = tk.Checkbutton(self, text="Enable Single Passage Mode", variable=var1, onvalue=1, offvalue=0, command=on_button_toggle1)
        checkbutton1.config(selectcolor="green", relief="raised")
        checkbutton1.grid(row=4, column=3, padx=10, pady=10)
        checkbutton1.flash()
        
        

        # setting up the ID entry box
        ID = tk.StringVar()
        IDEntry = tk.Entry(self,textvariable = ID)
        IDEntry.grid(row=1, column=3, padx=10, pady=10)

        # other notable labels
        IDLabel = ttk.Label(self, text ="Participant ID:")
        IDLabel.grid(row=1, column=2, padx=10, pady=10)
        Space = ttk.Label(self, text ="               ") # this is a spacer for formatting :)
        Space.grid(row=1, column=4, padx=10, pady=10)
        NoID = ttk.Label(self, text ="You need to include an ID before starting.")


# Moving to the actual apps...


# * Running Experiment Windows

class PSelectPg(tk.Frame):
    """
    The passage *set* selection window
    
    This also technically runs the experiment, but there is a different frame that shows the heading for running it.
    There's probably something I could do to neaten that up, but meh.
    """
    def __init__(self, parent, controller, debug):
        self.debug = debug
        self.controller = controller
        tk.Frame.__init__(self, parent)

        self.debug.initFrame(page=self,name="EsetSel")


        # Heading
        self.label = ttk.Label(self, text ="Select Set of Passages to Play:", font = LARGEFONT)
        self.label.grid(row = 0, column = 3, padx = 10, pady = 10)
        
        def focusbutton(): print("focus pressed")
        self.unfocus = ttk.Button(self, text="", command=focusbutton) # this button is purely for unfocusing the space bar
        self.binding = None

        # this doesn't work because Mac, but this is when a value is selected, the next button turns blue and is enabled
        # * correction, it does work on Mac, but the *blue button part* does not.
        def blueNext():
            self.nextB.config(style="BlueB.TButton",state = 'enabled')

        # Check Buttons
        v = tk.IntVar(self, 0) # Tkinter string variable able to store any string value -- THIS IS FOR THE CHECKBOXES
        # Dictionary to create multiple buttons -- These are the passage set options
        self.rb = [None, None, None, None, None, None, None, None]
        valuesL = {"Running Order 1" : 0,  # set as 0 & 1 for working with `passageOrders` but then +1 to them for viewing.
                "Running Order 2" : 1,
                "Running Order 3" : 2,  # set as 0 & 1 for working with `passageOrders` but then +1 to them for viewing.
                "Running Order 4" : 3}
        valuesR = {"Running Order 5" : 4,  
                "Running Order 6" : 5,
                "Running Order 7" : 6, 
                "Running Order 8" : 7}
        for i, (text, value) in enumerate(valuesL.items()): # creating check buttons
            self.rb[i] = tk.Radiobutton(self, text = text, variable = v, value = value, command=blueNext)
            self.rb[i].grid(row=value+1,column=2,padx=10,pady=10)
        
        for i, (text, value) in enumerate(valuesR.items()): # creating check buttons
            self.rb[i+4] = tk.Radiobutton(self, text = text, variable = v, value = value, command=blueNext)
            self.rb[i+4].grid(row=value-3,column=4,padx=10,pady=10)
        
        # left-over spacer label
        # Space = ttk.Label(self, text ="               ")
        # Space.grid(row=1, column=4, padx=10, pady=10)

        # Back Button
        self.backB = ttk.Button(self, text ="Back", command = lambda : controller.show_frame(StartPg))
        self.backB.grid(row = 10, column = 1, padx = 10, pady = 10)
  
        self.run, self.passageOrder, self.ID, self.passageNum = 0, 0, 0, 0
        self.volumes = np.zeros(shape=(2, 20), dtype=np.int_)
        self.preds = {
                0: {3:0,4:1,6:2,7:3,9:4,11:5,12:6,15:7},
                1: {1:0,2:1,5:2,8:3,10:4,13:5,14:6,16:7}
            }
        

        # this brings to next frame AND activates "space" to run the experiment.
        def nextF():
            """
            On next button, save passage set, and move to next window and activate binding to space
            """
            # stores passage order pick 
            passageSet = v.get() 
            # v.set(0)
            self.controller.shared_data["PassageOrder"].set(passageSet)

            # activates next frame and locks out any other tasks.
            self.controller.playingSet = True
            # controller.show_frame(RunExpPg) # showing next frame, tho this is kinda just a heading
            self.resetGrid()
            self.listenExplain()
            



        # Next Button
        self.nextB = ttk.Button(self, text ="Next", style="DisabledB.TButton", command=nextF, state = 'disabled')
        self.nextB.grid(row = 10, column = 5, padx = 10, pady = 10) # putting the button in its place by using grid

        self.heading = ttk.Label(self, text ="Whenever ready, press Space Bar to play the clip.", font = LARGEFONT)
        
        
        self.ready = ttk.Label(self, text="Ready for next.")

        self.vol = tk.StringVar()
        self.volEntry = tk.Entry(self,textvariable = self.vol)
        incorrectEntry = ttk.Label(self, text="Volume entered needs to be a number.")

        def trySubmit():
            if self.vol.get() == "": pass
            else:
                try:
                    t = int(self.vol.get())
                    self.playing()
                except:
                    incorrectEntry.grid(row=5,column=3,padx=10,pady=10)

                
            


        self.submitB = ttk.Button(self, text ="Submit Volume", command = trySubmit)

        self.exp = False
        self.lowVol, self.highVol = 0, 0
        self.expVolume = ttk.Label(self, text="XX", font = ("Verdana", 45))
        self.volHead = ttk.Label(self, text ="The volume for the following clips should be at this volume:", font = LARGEFONT)
        self.startExpB = ttk.Button(self, text ="Start", command = self.playing)
        self.moveExp = ttk.Button(self, text ="Submit Volume", command = self.finishListen)

        self.skipTrial = ttk.Button(self, text="Skip Trial", command=self.skipTri)
        self.skipVolume = ttk.Button(self, text="Skip Input", command=self.skipVol)
        self.skippedVol = False

        lExplaination = "This part of the experiment consists of listening to passages and adjusting the volume at the same time.\nYour goal is to lower the volume of the passages to the lowest possible volume that you can still understand the content of the passages.\n\nThis part of the experiment will determine the volume for the 2nd part of the experiment where you will be tasked with hearing new passages at a set volume and then repeating back the passage."
        self.lExplaination = ttk.Label(self, text=lExplaination)
        self.lExplainB = ttk.Button(self, text="Start", command=self.startListen)

        eExplaination = "This part of the experiment consists of listening to passages at a set volume that you just determined with the previous part.\nYour goal will be to retain and understand as much as you can from each passage, as after each passage you will be asked to repeat back the passage to the best of your ability."
        self.eExplaination = ttk.Label(self, text=eExplaination)
        self.eExplainB = ttk.Button(self, text="Start", command=self.startExp)

    def resetGrid(self):
        self.backB.grid_forget()
        self.label.grid_forget()
        for i in range(8): self.rb[i].grid_forget()
        self.nextB.grid_forget()


    def listenExplain(self):
        self.lExplaination.grid(row = 2, column= 3, padx = 10, pady = 10)
        self.lExplainB.grid(row = 3, column = 3, padx=10,pady=10)
    
    def skipTri(self):
        print("skipped trial", self.run)
        self.skipTrial.grid_forget() # don't know if this works well or not, it kinda makes mainloop have to do a fair bit of work
        self.skipTrial.grid(row = 4, column = 3, padx = 10, pady = 10)
        self.controller.focus()
        self.run += 1
        if self.run == 17 and False:
            self.heading.grid_forget()
            self.ready.grid_forget()
            self.skipTrial.grid_forget()
            self.expExplain()
        elif self.run == 21 and False:
            self.heading.grid_forget()
            self.ready.grid_forget()
            self.skipTrial.grid_forget()
            self.midExp()
        elif self.run == 42:
            self.finish()

    def skipVol(self):
        print("skipped volume", self.run-1)
        self.skippedVol = True
        if self.run < 42: self.playing()
        else: 
            self.heading.grid_forget()
            self.ready.grid_forget()
            self.volEntry.grid_forget()
            self.moveExp.grid_forget()
            self.skipVolume.grid_forget()
            self.expExplain()

    def startListen(self):
        self.playGrid()
        if self.controller.playingSet and not self.controller.playing: 
            self.playing()

    def playGrid(self):
        self.passageOrder = self.controller.shared_data["PassageOrder"].get()
        self.lExplaination.grid_forget()
        self.lExplainB.grid_forget()

        self.heading.grid(row = 0, column = 3, padx = 10, pady = 10)
        self.ready.grid(row = 3, column = 3, padx = 10, pady = 10)
        self.skipTrial.grid(row = 4, column = 3, padx = 10, pady = 10)

    def playing(self):
        if self.skippedVol:
            self.vol.set("")
            self.volEntry.grid_forget()
            self.submitB.grid_forget()
            self.skipVolume.grid_forget()
            self.skippedVol = False
        elif self.run > 1 and not self.exp:
            if self.passageNum < 21:
                # self.volumes, column 1: eye tracked, column 2: not eye tracked

                # y n  n  y  y  n  y  y n  y  n  y  y  n 
                # 9,21,29,15,14,27,18,2,35,13,39,12,10,31


                self.volumes[self.passageOrder, self.passageNum-1] = int(self.vol.get())
            else:
                self.volumes[0 if self.passageOrder == 1 else 1, self.passageNum-21] = int(self.vol.get())
            # running orders are opposites
            # 0 -> start high
            # 2, 2, 1, 2, 1, 1, 1, 2, 2, 1, 1
            # low –> 3,4,6,7,9,11,12,15
            # # high -> 1,2,5,8,10,13,14,16
            # if self.run in self.preds[1].keys():
            #     num = self.preds[1][self.run]
            # else:
            #     num = self.preds[0][self.run]
            # ee = self.vol.get()
            # self.volumes[pred, num] = int(self.vol.get()) # when RO = 0, high first, then low. When RO = 1, low first, then high.
            self.vol.set("")
            self.volEntry.grid_forget()
            self.skipVolume.grid_forget()
            self.submitB.grid_forget()
        # elif self.exp and (self.run == 17 or self.run == 21):
        #     print("starting experimental listening")
        #     self.expVolume.grid_forget()
        #     self.volHead.grid_forget()
        #     self.startExpB.grid_forget()
        #     self.heading.grid(row = 0, column = 3, padx = 10, pady = 10)
        self.ready.grid(row = 3, column = 3, padx = 10, pady = 10)
        self.skipTrial.grid(row = 4, column = 3, padx = 10, pady = 10)
        self.binding = self.controller.bind("<space>", self.key)
        self.spaceBound = True

    def key(self,event):
        """This program, on space key being pressed, plays a sound clip and records from the camera.

        :param event: tk event: `<space>` button pressed
        :type event: tk.event
        """
        # Setting up base important variables.
        print("\n\n\nKEY: playing", self.run, "\n\n\n")
        if self.run < 41:
            self.ready.grid_forget()
            self.skipTrial.grid_forget()
            self.controller.readyPlay.set(False)
            self.controller.playing, self.ID, self.passageOrder = True, self.controller.shared_data["ID"].get(), self.controller.shared_data["PassageOrder"].get()
            self.passageNum = runningOrders[self.passageOrder][self.run] # listenOrders[Orders[self.passageOrder][0]][self.run] # sets the passage number based on the passage order and current run number.
            # Getting the audio file.
            PATH = os.path.join(base_dir, f"NewPilotStimuli/P{self.passageNum}.wav")
            passage = AudioSegment.from_wav(PATH)

            # Helpful for telling debugging that successfully running multiple passages
            print("running multiple passages") # TODO: For Final, cleanup this print statement from final prob.

            self.queue = queue.Queue()
            ThreadedTask(self, self.queue, passage).start()
            self.master.after(100, self.process_queue)

            
            # After successful recording, increase run count for next recording.
            self.run +=1
            self.controller.playing = False
            self.controller.readyPlay.set(True)
        elif self.run > 16 and self.run < 25 and False:
            self.ready.grid_forget()
            self.skipTrial.grid_forget()
            self.controller.readyPlay.set(False)
            self.controller.playing, self.ID, self.passageOrder = True, self.controller.shared_data["ID"].get(), self.controller.shared_data["PassageOrder"].get()
            self.passageNum = expOrders[Orders[self.passageOrder][1]][self.run - 17] # sets the passage number based on the passage order and current run number.
            # Getting the audio file.
            PATH = os.path.join(base_dir, f"PilotStimuli/P{self.passageNum}.wav")
            passage = AudioSegment.from_wav(PATH)


            print("running experimental passages")
            self.queue = queue.Queue()
            ExpThreadedTask(self, self.queue, passage).start()
            self.master.after(100, self.process_queue)

            
            # After successful play, increase run count for next play.
            self.run +=1
            self.controller.playing = False
            self.controller.readyPlay.set(True)
        else:
            self.finish()

    def play(self, passage):
        # print("\n\n\nPLAY: playing", self.run, "\n\n\n")
        cap = cv2.VideoCapture(0) # ! THIS NEEDS TO CHANGE BASED ON COMPUTER CAMERA LAYOUT. TEST BEFORE EACH NEW DEVICE RUNNING 
        # ! My Macbook treats the webcam as camera 0. I do not know what yours might treat it as. 
        # If this does in fact matter, I can set up a menu "configure camera" button that shows different cameras.
        # It would just grab from the `adjustCamPg` and try work from there. 
        cap.set(cv2.CAP_PROP_FPS, 42)
        # Defining specifications for the video recording.
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) # Width/Height of the camera.
        codec = cv2.VideoWriter_fourcc('m','p','4','v') # This is the mp4 codec.
        recording_flag, frames = True, 0  # we are transitioning from not recording to recording

        t = threading.Thread(target=play, args=(passage,))# Setting up the threading for successful playing of the audio.

        # Beginning recording
        outPATH = os.path.join(base_dir, f"Video Files/P{self.ID}/Order{self.passageOrder}Test{self.run}.mp4")
        output = cv2.VideoWriter(outPATH, codec, 30, size,1) 

        # fps = cap.get(cv2.CAP_PROP_FPS)
        # frameref_ms = int(time.time()*1000)
        # frametime_ms = int(1000/fps)
        # frameCounter = 0
        # frameTimes = []
        # frameCounts = []

        # self.after((passage.duration_seconds*1000), self.watchforPlay)

        # print(f"fps: {cap.get(cv2.CAP_PROP_FPS)}")
        # print(float(passage.duration_seconds))
        
        # frameArray = []

        t.start() # Begin audio at the same time as beginning recording.
        # prevTime = time.time()
        while True: 
            if frames >= (30*(passage.duration_seconds)) or not recording_flag: # stop recording once time has ended
                # print(cap.get(cv2.CAP_PROP_FPS),20,passage.duration_seconds,frames,cv2.CAP_PROP_FPS)
                # ! This differs based on *camera*, so if the camera has a recording FPS that's not 30, this will need to be changed.
                # Currently, it records for 30frames * seconds of the audio file, which is designed for a 30 frames per second camera.
                print("stopping recording") # TODO: For Final, cleanup print
                
                # Stopping recording and releasing the output to be saved.
                output.release()
                cap.release() 
                break # breaking out of the loop
            # print(f"fps: {cap.get(cv2.CAP_PROP_FPS)}")
            # Reading and Writing Frames
            # frameref_ms += frametime_ms

            ret, frame = cap.read() 
            if ret == True: output.write(frame)
            else: print("end?")

            # currTime = time.time()
            # timeDiff = currTime-prevTime
            # print("\n\n",prevTime,currTime,"\n\n")
            # if int(prevTime) < int(currTime): 
            #     frameCounts.append(frameCounter)
            #     frameCounter = 1
            # else: frameCounter +=1
            # print("\n\n",(timeDiff*1000), "\n",frametime_ms, "\n",frameref_ms,"\n",frameCounter,"\n\n")
            # frameTimes.append((timeDiff*1000))
            frames +=1
            # prevTime = currTime
            # time.sleep(0.001667)
            
            # time.sleep((int(time.time()*1000)-frameref_ms))

        # print("\n"+str(frames/30)+"\n\n\n")
        # print(statistics.mean(frameTimes),statistics.mean(frameCounts))

    def process_queue(self):
        if not self.queue.empty():
            print(self.run) # Once all 17 listening passages (including practice) have been played, move to next page.
            if self.run <= 41: 
                if self.run == 1: self.playing()
                else: self.confirmVol()
            elif self.run == 21 and False:
                self.midExp()
            elif self.run < 25 and False:
                self.playing()
            else:
                self.finish()
        else:
            if self.spaceBound: 
                self.controller.unbind("<space>", self.binding)
                self.spaceBound = False
            self.master.after(100, self.process_queue)

    def confirmVol(self):
        if self.spaceBound: 
            self.controller.unbind("<space>", self.binding)
            self.spaceBound = False
        self.volEntry.grid(row = 3, column = 3, padx = 10, pady = 10)
        self.skipVolume.grid(row = 5, column = 3, padx = 10, pady = 10)
        if self.run < 17: self.submitB.grid(row = 4, column = 3, padx = 10, pady = 10)
        else: self.moveExp.grid(row = 4, column = 3, padx = 10, pady = 10)

    def finishListen(self):
        print("finish listen")
        if self.vol.get() == "": print("vol returned empty")
        self.volumes[(0 if self.passageNum < 9 else 1),7] = int(self.vol.get())
        self.vol.set("")
        print("done") # TODO: For Final, cleanup print
        self.controller.playingSet = False
        
        self.heading.grid_forget()
        self.ready.grid_forget()
        self.volEntry.grid_forget()
        self.moveExp.grid_forget()
        self.skipVolume.grid_forget()
        
        self.expExplain()


    def expExplain(self):
        self.eExplaination.grid(row = 2, column= 3, padx = 10, pady = 10)
        self.eExplainB.grid(row = 3, column = 3, padx=10,pady=10)
        self.volumes.sort()
        setVolHigh, setVolLow = calcMean(self.volumes[0]),calcMean(self.volumes[1])
        # 1 & 2 start with high pred first. 3 & 4 start with low first
        if self.passageOrder in [0, 1, 6, 7]: 
            self.expVolume.config(text=str(setVolHigh))
            self.OtherVol = setVolLow
        else: 
            self.expVolume.config(text=str(setVolLow))
            self.OtherVol = setVolHigh

    def startExp(self):
        self.eExplaination.grid_forget()
        self.eExplainB.grid_forget()

        self.exp = True
        self.volHead.grid(row = 0, column = 5, padx = 10, pady = 10)
        self.startExpB.grid(row=2,column=5,padx=10,pady=10)
        
        
        self.expVolume.grid(row = 1, column = 5, padx = 10, pady = 10)
    
    def midExp(self):
        self.heading.grid_forget()
        self.ready.grid_forget()

        # TODO: this is being funky 
        self.volHead.grid(row = 0, column = 3, padx = 10, pady = 10)
        self.startExpB.grid(row=2, column = 3, padx=10,pady=10)

        # 1 & 2 start with high pred first. 3 & 4 start with low first
        self.expVolume.config(text=str(self.OtherVol))
        self.expVolume.grid(row = 1, column = 3, padx = 10, pady = 10)


    
    def finish(self):
        self.controller.show_frame(TestComplete)



    
        


class ThreadedTask(threading.Thread):
        # class to go with passage playing, -- updates frame whilst playing audio and recording
        def __init__(self, page, queue1, passage):
            super().__init__()
            self.queue = queue1
            self.page = page
            self.passage = passage
        def run(self):
            self.page.play(self.passage)
            self.queue.put("Task finished")

class ExpThreadedTask(threading.Thread):
        # class to go with passage playing, -- updates frame whilst playing audio
        def __init__(self, page, queue1, passage):
            super().__init__()
            self.queue = queue1
            self.page = page
            self.passage = passage
        def run(self):
            play(self.passage)
            self.queue.put("Task finished")

class RunExpPg(tk.Frame):
    """
    no longer needed I think


    XXX: remove
    """
    def __init__(self, parent, controller, debug):
        self.controller = controller
        self.debug = debug
        tk.Frame.__init__(self, parent)

        # Heading
        self.label = ttk.Label(self, text ="Whenever ready, press Space Bar to play the clip.", font = LARGEFONT)
        self.label.grid(row = 0, column = 5, padx = 10, pady = 10)
        
        self.controller.playing = False

        self.ready = ttk.Label(self, text="Ready for next.")
        self.ready.grid(row = 3, column = 5, padx = 10, pady = 10)

        # if not self.controller.playing:
        #     self.load()
        # else:
        #     self.ready.grid(row = 3, column = 5, padx = 10, pady = 10)
        #     print('Updated')

        if self.controller.readyPlay.get():
            self.load()
        else:
            self.ready.grid(row = 3, column = 5, padx = 10, pady = 10)


    def load(self):
        self.ready.grid_forget()
        print('Updated')

class OnePSelectPg(tk.Frame):
    """
    The *single*-passage selection window.
    Same as set passage selection, but only for a single passage.

    Comments here may be less helpful than comments in `PSelectPg` due to these two being very similar.
    Realistically, I could have condensed them down into the same function, but it would not have been as readable and understandable, at least I think.



    TODO: fix up to what set passage does
    """
    def __init__(self, parent, controller, debug):
        self.controller = controller
        self.debug = debug
        tk.Frame.__init__(self, parent)

        self.debug.initFrame(page=self,name="EpSel")

        # Heading
        label = ttk.Label(self, text ="Select Passage to Play:", font = LARGEFONT)
        label.grid(row = 0, column = 3, padx = 10, pady = 10)

        # this doesn't work because Mac, but this is when a value is selected, the next button turns blue and is enabled
        # yeyeye it does kinda work, see same funct in `PSelectPg`
        def blueNext():
            button2.config(style="BlueB.TButton",state = 'enabled')

        # Passage select checkboxes
        v = tk.IntVar(self, 10) # This is the value of the passage checkboxes, set at 10 (or any non-existent value) to not have any selected by default.
        # Dictionaries to create multiple buttons -- These are the passage set options
        valuesL = {"Passage 1" : 1, 
                "Passage 2" : 2,
                "Passage 3" : 3,
                "Passage 4" : 4}
        valuesR = {"Passage 5" : 5,
                "Passage 6" : 6,
                "Passage 7" : 7,
                "Passage 8" : 8}
        for (text, value) in valuesL.items(): # creating check buttons, left row
            tk.Radiobutton(self, text = text, variable = v, value = value, command=blueNext).grid(row=int(value+1),column=2,padx=10,pady=10)
        for (text, value) in valuesR.items(): # creating check buttons, right row
            tk.Radiobutton(self, text = text, variable = v, value = value, command=blueNext).grid(row=int(value-3),column=4,padx=10,pady=10)
        tk.Radiobutton(self, text = "Passage 0", variable = v, value = 0, command=blueNext).grid(row=int(1),column=3,padx=10,pady=10) # creating check buttons, middle row, only passage 0


        # Back Button
        button1 = ttk.Button(self, text ="Back", command = lambda : controller.show_frame(StartPg))
        button1.grid(row = 10, column = 1, padx = 10, pady = 10)
  
        self.controller.playing, self.run, self.passageOrder, self.ID, self.passageNum = False, 0, 0, 0, 0

        # this brings to next frame
        def nextF():
            """
            On next button, save passage set, and move to next window and running passages
            """
            # stores passage order pick 
            passage = v.get() 
            self.controller.shared_data["Passage"].set(passage)

            # Recording time
            self.controller.playingOne = True
            controller.show_frame(Run1PassagePg)
            if self.controller.playingOne and not self.controller.playing: self.controller.bind("<space>", self.key)

        # next button
        button2 = ttk.Button(self, text ="Next", style="DisabledB.TButton", command=nextF, state = 'disabled')
        button2.grid(row = 10, column = 5, padx = 10, pady = 10) # putting the button in its place by using grid
    
    def key(self,event):
        """See same funct in `PSelectPg`

        :param event: tk event
        :type event: tk event
        """
        self.controller.playing, self.ID, self.passageOrder = True, self.controller.shared_data["ID"].get(), self.controller.shared_data["PassageOrder"].get()
        self.passageNum = self.controller.shared_data["Passage"].get()

        # Audio
        PATH = os.path.join(base_dir, f"PilotStimuli/P{self.passageNum}.wav")
        passage = AudioSegment.from_wav(PATH)

        print("running 1 passage") # TODO: For Final, cleanup print

        # video capture
        cap = cv2.VideoCapture(0) 

        # video metrics
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        codec = cv2.VideoWriter_fourcc('m','p','4','v')

        frames = 0 # we are transitioning from not recording to recording

        # multithreading for audio & video
        t = threading.Thread(target=play, args=(passage,))
        
        # recording
        outPATH = os.path.join(base_dir, f"Video Files/P{self.ID}/Passage {self.passageNum}.mp4")
        output = cv2.VideoWriter(outPATH, codec, 30, size,1) 

        t.start() # start threading

        while True: 
            # see `PSelectPg` for more info
            if frames >= (30*math.ceil(float(passage.duration_seconds))):
                output.release()
                cap.release() 
                cv2.destroyAllWindows()
                break
            ret, frame = cap.read() 
            output.write(frame)
            frames +=1
        
        self.controller.playing = False
        self.controller.playingOne = False
        # since only one passage, after recording move to test complete
        self.controller.show_frame(TestComplete)

class Run1PassagePg(tk.Frame):
    """
    Beginning to run the experiment (for 1 passage)
    """
    def __init__(self, parent, controller, debug):
        self.controller = controller
        self.debug = debug
        tk.Frame.__init__(self, parent)

        # Heading
        self.label = ttk.Label(self, text ="Whenever ready, press Space Bar to play the clip.", font = LARGEFONT)
        self.label.grid(row = 0, column = 5, padx = 10, pady = 10)

class TestComplete(tk.Frame):
    """
    TEST COMPLETED! window
    this is the end
    """
    def __init__(self, parent, controller, debug):
        self.controller = controller
        self.debug = debug
        tk.Frame.__init__(self, parent)

        # Heading
        label = ttk.Label(self, text ="Tests Completed.", font = LARGEFONT)
        label.grid(row = 0, column = 5, padx = 10, pady = 10)


# removed this functionality
class CamAdjustPg(tk.Frame):
    """
    Old adjust camera page.
    This funtionality was removed due to many issues in application form.
    Has useful code so i am hesitant to get rid of it.
    TODO: For Final, remove in final
    """
    def __init__(self, parent, controller, debug):
        self.controller = controller
        self.debug = debug
        tk.Frame.__init__(self, parent)

        # Heading
        label = ttk.Label(self, text ="Adjust Camera to Green Cross in Bottom Left Corner", font = LARGEFONT)
        label.grid(row = 0, column = 2, padx = 10, pady = 10)

        label_widget = tk.Label(self) 
        label_widget.grid(row=1, column=2, padx=5,pady=5) 
        self.adjust, self.cap, self.done = 0, None, False

        def camera():
            _, frame = self.cap.read() 
  
            opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA) 
            captured_image = Image.fromarray(opencv_image) 
            photo_image = ImageTk.PhotoImage(image=captured_image) 
        
            # Displaying photoimage in the label 
            label_widget.photo_image = photo_image 
            label_widget.configure(image=photo_image) 
        
            # Repeat the same process after every 10 seconds 
            if not self.done: label_widget.after(10, camera) 

        def adjust():
            self.cap = cv2.VideoCapture(1)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))/2) 
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))/2) 
            button1.grid_forget()
            label_widget.grid(row=1, column=2, padx=5,pady=5) 
            self.adjust += 1
            nextb.config(style="BlueB.TButton",state="normal")
            camera()
            

        def done():
            self.done = True
            self.cap.release()
            controller.show_frame(Run1PassagePg if singlePassage else RunExpPg)

        def back():
            if self.adjust > 0: 
                self.done = True
                label_widget.grid_forget()
                button1.grid(row = 2, column = 2, padx = 10, pady = 10)
                nextb.config(style="DisabledB.TButton",state="disabled")
                self.cap.release()
                self.adjust = 0
                self.cap = None
            controller.show_frame(OnePSelectPg if singlePassage else PSelectPg)

        ## Start button
        button1 = ttk.Button(self, text ="Adjust Camera", command = adjust)
        button1.grid(row = 2, column = 2, padx = 10, pady = 10)

         # Back Button
        global singlePassage
        back = ttk.Button(self, text ="Back", command = back)
        back.grid(row = 10, column = 1, padx = 10, pady = 10)

        nextb = ttk.Button(self, text ="Finish", command = done,style="DisabledB.TButton",state="disabled")
        nextb.grid(row = 10, column = 3, padx = 10, pady = 10)














# * DIFFERENT TYPES OF APPLICATIONS IN THE SAME APP WOAHHHH 
# * wow i should have made two apps














# Analysing Data Windows

class PassagePg(tk.Frame):
    """
    The select *single* passage window
    """
    def __init__(self, parent, controller, debug):
        self.controller = controller
        self.debug = debug
        tk.Frame.__init__(self, parent)

        # Heading
        label = ttk.Label(self, text ="Select Passage:", font = LARGEFONT)
        label.grid(row = 0, column = 3, padx = 10, pady = 10)

        # this doesn't work because Mac, but this is when a value is selected, the next button turns blue and is enabled
        # no, see `PSelectPg` for more
        def blueNext():
            button2.config(style="BlueB.TButton",state = 'enabled')

        # Passages checkboxes 
        v = tk.IntVar(self, 20) # see `OnePSelectPg` for more info on this
        # Dictionaries to create multiple buttons -- These are the passage options
        valuesL = {"Passage 1" : 1, 
                "Passage 2" : 2,
                "Passage 3" : 3,
                "Passage 4" : 4,
                "Passage 5" : 5, 
                "Passage 6" : 6,
                "Passage 7" : 7,
                "Passage 8" : 8}
        valuesR = {"Passage 9" : 9,
                "Passage 10" : 10,
                "Passage 11" : 11,
                "Passage 12" : 12,
                "Passage 13" : 13,
                "Passage 14" : 14,
                "Passage 15" : 15,
                "Passage 16" : 16}
        for (text, value) in valuesL.items(): # creating check buttons, left side
            tk.Radiobutton(self, text = text, variable = v, value = value, command=blueNext).grid(row=int(value+1),column=2,padx=10,pady=10)
        for (text, value) in valuesR.items(): # creating check buttons, right side
            tk.Radiobutton(self, text = text, variable = v, value = value, command=blueNext).grid(row=int(value-7),column=4,padx=10,pady=10)
        tk.Radiobutton(self, text = "Passage 0", variable = v, value = 0, command=blueNext).grid(row=int(1),column=3,padx=10,pady=10) # creating check buttons, middle, only passage 0

        # Back Button
        button1 = ttk.Button(self, text ="Back", command = lambda : controller.show_frame(StartPg))
        button1.grid(row = 10, column = 1, padx = 10, pady = 10)
  
        # this brings to upload page
        def nextF():
            """
            On next, save passage number and move to next window (upload)
            """
            passage = v.get() # stores passage pick 
            self.controller.shared_data["Passage"].set(passage)
            controller.show_frame(UploadPg)

        # Next Button
        button2 = ttk.Button(self, text ="Next", style="DisabledB.TButton", command=nextF, state = 'disabled')
        button2.grid(row = 10, column = 5, padx = 10, pady = 10)

class PassageSetPg(tk.Frame):
    """
    The passage *set* selection window
    Effectively the same as `PassagePg`, but follows the layout of `PSelectPg`.
    """    
    def __init__(self, parent, controller, debug):
        self.controller = controller
        self.debug = debug
        tk.Frame.__init__(self, parent)

        # Heading
        label = ttk.Label(self, text ="Select Set of Passages:", font = LARGEFONT)
        label.grid(row = 0, column = 2, padx = 10, pady = 10)

        # `PSelectPg` more info
        def blueNext():
            button2.config(style="BlueB.TButton",state = 'enabled')

        # checkboxes
        v = tk.StringVar(self, " ") 
        values = {"Running Order 1" : 0, # Dictionary to create multiple buttons -- These are the passage set options
                "Running Order 2" : 1}
        for (text, value) in values.items(): # creating check buttons
            tk.Radiobutton(self, text = text, variable = v, value = value, command=blueNext).grid(row=value+1,column=2,padx=10,pady=10)
        
        # Back Button
        button1 = ttk.Button(self, text ="Back", command = lambda : controller.show_frame(StartPg))
        button1.grid(row = 10, column = 1, padx = 10, pady = 10)
  
        # this brings to next frame upload
        def nextF():
            """
            On next button, save passage set, and move to next window upload
            """
            passage = v.get() # stores passage pick 
            self.controller.shared_data["PassageOrder"].set(passage)
            controller.show_frame(UploadPg)

        # next button
        button2 = ttk.Button(self, text ="Next", style="DisabledB.TButton", command=nextF, state = 'disabled')
        button2.grid(row = 10, column = 3, padx = 10, pady = 10) # putting the button in its place by using grid

class UploadPg(tk.Frame): 
    """
    The Upload Video page

    Differs based on `singlePassage`.
    """
    def __init__(self, parent, controller, debug):
        self.controller = controller
        self.debug = debug
        tk.Frame.__init__(self, parent)

        # Heading
        label = ttk.Label(self, text ="Upload the Video", font = LARGEFONT)
        label.grid(row = 0, column = 2, padx = 10, pady = 10)
  
        # Back Button 
        #TODO: currrently doesn't reset the video, and therefore can't get change video
        button1 = ttk.Button(self, text ="Back", command = lambda : controller.show_frame(PassagePg if singlePassage else PassageSetPg))
        button1.grid(row = 10, column = 1, padx = 10, pady = 10)

       

        # Next Frame Function
        def nextF():
            """
            On next, move to next crop page.
            """
            global singlePassage
            controller.show_frame(cropPg)
            button2.config(style="DisabledB.TButton", state = 'disabled')
        
        # Next Button
        button2 = ttk.Button(self, text ="Next", style="DisabledB.TButton", command=nextF, state = 'disabled')
        button2.grid(row = 10, column = 3, padx = 10, pady = 10)

        # On upload function
        def UploadAction():
            global singlePassage, files
            
            if singlePassage: self.controller.shared_data["video"].set(filedialog.askopenfilename()) # stores filename if singlePassage
            else: 
                while len(files) != 17: # TODO: For final, fix
                    files = filedialog.askopenfilenames() # if not, open multiple files (multiple videos)
                    if len(files) !=17: tk.messagebox.showwarning(title="Uploaded Files Warning", message=f"Incorrect number of files uploaded. \nShould be 17 files, instead {len(files)} uploaded.\n\nPlease upload the correct number of files.") # Too few/many files message box


            # stores whether video or not
            self.controller.settings.entryconfigure("Set Image Crop", state=tk.NORMAL) # ! also a relic, this allows the settings menu to crop now, although crop is default now.
            button2.config(style="BlueB.TButton",state = 'enabled') # enables next button
    
        # Upload Button
        uploadB = tk.Button(self, text='Upload', command=UploadAction)
        uploadB.grid(row = 2, column = 2, padx = 10, pady = 10)

class WorkingPg(tk.Frame):
    """
    The parsing of *individual* passages window. This is where the magic happens, or something.
    """
    def __init__(self, parent, controller, debug):
        self.controller = controller
        self.debug = debug
        tk.Frame.__init__(self, parent)
        
        # Heading
        label = ttk.Label(self, text ="Video Processing", font = LARGEFONT)
        label.grid(row = 0, column = 2, padx = 10, pady = 10)

        # Back Button
        button1 = ttk.Button(self, text ="Back", command = lambda : controller.show_frame(PassagePg))
        button1.grid(row = 10, column = 1, padx = 10, pady = 10)

        # Progress Bar
        self.progress = ttk.Progressbar(self, orient = tk.HORIZONTAL, length=100, maximum=100, mode = 'determinate') 
        self.progress.grid(row=3,column=2,padx=10,pady=10)
        self.barVal = 0.0
        self.emptyTime = 0
        self.p = 1 
        
    
        def bar():
            print("\n\nbar\n\n")
            startB.grid_forget()
            label.config(text= "Video Now Processing") # supposed to tell user that it's actually working but :shrug:

            # Initialising base variables from shared data.
            passage = self.controller.shared_data["Passage"].get() # grabs passage from shared_data

            cropList = self.controller.shared_data["crop"].get().split() # grabs crop listing from shared data
            cropBounds = {"H":[cropList[0],cropList[1]],
                          "W":[cropList[2],cropList[3]]}
            y1, y2, x1, x2 = int(cropBounds["H"][0]), int(cropBounds["H"][1]), int(cropBounds["W"][0]), int(cropBounds["W"][1]) # Defining crop based on cropBounds

            videoFile = self.controller.shared_data["video"].get() # setting video file path 

            # TODO: might be worthwhile to give estimated processing time and/or estimated end time
            count, self.res, self.flags, self.batchedFlags = 0, [], [], {} # important variables to this window

            timeIt.Start("video capture cv")
            video = cv2.VideoCapture(videoFile) # grabbing video video.get(cv2.CAP_PROP_FRAME_COUNT)


            shape = (int(video.get(cv2.CAP_PROP_FRAME_COUNT)),(y2-y1),(x2-x1),3)
            self.frames = np.empty(shape, dtype="uint8")            
            success,image = video.read()
            
            self.queue = queue.Queue()
            ThreadedLoadingBar(self, self.queue, cropBounds, video, success, image, passage).start()
            print("\n\nqueue\n\n")
            if self.emptyTime < 800: self.master.after(1, self.process_queue)

            self.queue.join()


            timeIt.Stop()
            # self.res, self.flags, self.batchedFlags = parse(self.frames, cropBounds)
            if False: # * using this to hide useful comments for later
                pass
                # * come back to this later if needed
                # timeIt.Start("video capture imgio 1")

                # # vid = iio.get_reader(videoFile,  'ffmpeg')
                # # # number of frames in video
                # # frameNum = vid.count_frames()
                
                # meta = iio.improps(videoFile, plugin="pyav").shape[0]
                # self.frames1 = np.empty((meta), dtype=np.ndarray) # TODO, maybe restack this after grabbing all frames?
                # # self.frames1 = np.empty([int(frameNum)], dtype=np.ndarray) # TODO, maybe restack this after grabbing all frames?
                # for i, frame in enumerate(iio.imiter(videoFile, plugin="pyav",format="rgb24", thread_type="FRAME")):
                #     self.frames1[i] = frame[y1:y2,x1:x2]

                # timeIt.Stop()

                # timeIt.Start("video capture imgio 2")

                # vid = iio.get_reader(videoFile,  'ffmpeg')
                # # number of frames in video
                # frameNum = vid.count_frames()
                # self.frames2 = iio.imread(videoFile, plugin="pyav")

                # timeIt.Stop()
                # changeIndex = [(i, y) for i, y in enumerate(self.frames) if y.all() != 0]

            

        # Start button
        startB = ttk.Button(self, text="Start Processing", command = bar)
        startB.grid(row=2, column=2, padx=10, pady=10)

        # Spacer for symmetry
        Space = ttk.Label(self, text ="          ")
        Space.grid(row=1, column=3, padx=10, pady=10)

    def getFrames(self, i, video, y1,y2,x1,x2, success, image, prev_frame_downscaled):
        if not success: 
            return("broke")
        frame = image[y1:y2,x1:x2] # adding each frame into a list

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # greyscale
        blur = cv2.GaussianBlur(grey, (5,5), 0) # blurred
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] # this makes the contours pop out
        invert = 255 - thresh
        if i > 0: 
            diff_index = np.sum(np.abs(invert - prev_frame_downscaled))
            if diff_index > 40000: self.frames[i] = frame
        else:
            self.frames[i] = frame
        prev_frame_downscaled = invert

        success, image = video.read()
        return success, image, prev_frame_downscaled

    def parse(self, passage, y1, y2, x1, x2):
        self.res, self.flags, self.fflags, self.bF = threadedParse(self.frames, ((y2-y1),(x2-x1),3), self)
        global fPrecursor, rPrecursor, ffPrecursor, bfPrecursor, TfPrecursor, TrPrecursor, TffPrecursor, TbfPrecursor
        fPrecursor[passage], rPrecursor[passage], ffPrecursor[passage], bfPrecursor[passage] = self.flags, self.res, self.fflags, self.bF # Storing data for global use
        # TfPrecursor[passage], TrPrecursor[passage], TffPrecursor[passage], TbfPrecursor[passage] = tflags, tres, tfflags, tbF    
    
    def process_queue(self):
        # print("\n\nqueue test\n\n")
        if round(self.barVal) >= 100 and self.queue.empty() or self.emptyTime == 800 and self.queue.empty(): 
            if self.barVal < 100: self.progress.step(100-self.barVal)
            if self.emptyTime<1000: print(self.emptyTime)
            self.controller.show_frame(FinalPg) 
            self.queue.task_done()
        else:
            if not self.queue.empty():
                # print("got")
                self.emptyTime = 0
                step = self.queue.get()
                self.progress.step(float(step))
                self.barVal += float(step)
                print("got", step, "- full:", self.barVal)

                
                self.master.after(1, self.process_queue)
            else:
                self.emptyTime += 1
                # print("\n\n\n\nqueue empty\n\n\n\n")
                self.master.after(1, self.process_queue)

class WorkingSetPg(tk.Frame):
    """
    The parsing of *set* of passages window
    """
    def __init__(self, parent, controller, debug):
        self.controller = controller
        self.debug = debug
        tk.Frame.__init__(self, parent)

        # Heading
        label = ttk.Label(self, text ="Video Processing", font = LARGEFONT)
        label.grid(row = 0, column = 2, padx = 10, pady = 10)

        # Back Button
        button1 = ttk.Button(self, text ="Back", command = lambda : controller.show_frame(PassageSetPg))
        button1.grid(row = 10, column = 1, padx = 10, pady = 10)

        # Progress Bar
        self.progress = ttk.Progressbar(self, orient = tk.HORIZONTAL, length=100, maximum=100, mode = 'determinate') 
        self.progress.grid(row=3,column=2,padx=10,pady=10)
        self.barVal = 0.0
        self.emptyTime = 0

        self.lenFrames = 0
        self.index = 0


        def bar():
            # Grabbing key info from shared data
            print("\n\nbar\n\n")
            startB.grid_forget()
            label.config(text= "Video Now Processing") # supposed to tell user that it's actually working but :shrug:
            self.emptyTime = 0
            passageOrder = self.controller.shared_data["PassageOrder"].get() # Passage order

            cropList = self.controller.shared_data["crop"].get().split() # Crop bounds
            self.cropBounds = {"H":[cropList[0],cropList[1]],
                          "W":[cropList[2],cropList[3]]}
            y1, y2, x1, x2 = int(self.cropBounds["H"][0]), int(self.cropBounds["H"][1]), int(self.cropBounds["W"][0]), int(self.cropBounds["W"][1]) # Defining crop based on cropBounds
            self.queue = queue.Queue()
            # differentiating based on passage here
            global listenOrders, files
            for index in range(17):
                passage = listenOrders[passageOrder][index]
                self.p = index+1
                # for index, passage in enumerate(passageOrders[passageOrder]):
                videoFile = files[index] # Determines video file from global files list.

                video = cv2.VideoCapture(videoFile) # grabbing video video.get(cv2.CAP_PROP_FRAME_COUNT)

                shape = (int(video.get(cv2.CAP_PROP_FRAME_COUNT)),(y2-y1),(x2-x1),3)
                self.frames = np.empty(shape, dtype="uint8")            
                success,image = video.read()
            
                numFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                self.lenFrames = numFrames
                frequency = numFrames//30
                vidCapStep = frequency*((100/17)/3)/numFrames
                remFrames = numFrames%frequency
                print("\n\nFRAMES REM:", remFrames, "\n\n")
                prev_frame_downscaled = 0

                for i in range(numFrames):
                    success, image, prev_frame_downscaled = self.getFrames(i, video, y1, y2, x1, x2, success, image, prev_frame_downscaled)
                    if i % frequency == 0 and i != 0: 
                    #     self.page.queue.put(vidCapStep)
                        self.barVal += vidCapStep
                        print("frames - put", vidCapStep, "- total val:", self.barVal)
                self.barVal += remFrames*((100/17)/3)/numFrames
                print("last frames - put", remFrames*((100/17)/3)/numFrames, "- total val:", self.barVal)
                print("Frames Correct:", round((self.barVal/self.p),10) == round((100/17)/3,10))
                self.parse(passage, y1, y2, x1, x2)
                print("\n\n\ndone:", index,"\n\n\n")

            print("done all.")











            controller.show_frame(FinalPg)


            # if self.emptyTime < 800: self.master.after(1, self.process_queue)

            # while self.index < 8:
            #     self.index += 1
            #     bar()

            # self.queue.join()
            # print("joined")
            


        # Start processing button
        startB = ttk.Button(self, text="Start Processing", command = bar)
        startB.grid(row=2, column=2, padx=10, pady=10)

        # spacer for nice gui :)
        Space = ttk.Label(self, text ="          ")
        Space.grid(row=1, column=3, padx=10, pady=10)

    def getFrames(self, i, video, y1,y2,x1,x2, success, image, prev_frame_downscaled):
        if not success: 
            return("broke")
        frame = image[y1:y2,x1:x2] # adding each frame into a list

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # greyscale
        blur = cv2.GaussianBlur(grey, (5,5), 0) # blurred
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] # this makes the contours pop out
        invert = 255 - thresh
        if i > 0: 
            diff_index = np.sum(np.abs(invert - prev_frame_downscaled))
            if diff_index > 40000: self.frames[i] = frame
        else:
            self.frames[i] = frame
        prev_frame_downscaled = invert

        success, image = video.read()
        return success, image, prev_frame_downscaled

    def parse(self, passage, y1, y2, x1, x2):
        self.res, self.flags, self.fflags, self.bF = threadedParse(self.frames, ((y2-y1),(x2-x1),3), self)
        global fPrecursor, rPrecursor, ffPrecursor, bfPrecursor, TfPrecursor, TrPrecursor, TffPrecursor, TbfPrecursor
        fPrecursor[passage], rPrecursor[passage], ffPrecursor[passage], bfPrecursor[passage] = self.flags, self.res, self.fflags, self.bF # Storing data for global use
        # TfPrecursor[passage], TrPrecursor[passage], TffPrecursor[passage], TbfPrecursor[passage] = tflags, tres, tfflags, tbF    
    
    def process_queue(self):
        if round(self.barVal) >= 100 and self.queue.empty() or self.emptyTime == 800 and self.queue.empty(): 
            self.emptyTime = 800
            print("done")
            if self.index == 17: 
                if self.barVal < 100: self.progress.step(100-self.barVal)
                self.controller.show_frame(FinalPg) 
            self.queue.task_done()
        else:
            if not self.queue.empty():
                # print("got")
                self.emptyTime = 0
                step = self.queue.get()
                self.progress.step(float(step)/17)
                self.barVal += float(step)/17
                print("got", step, "- full:", self.barVal)

                
                self.master.after(1, self.process_queue)
            else:
                self.emptyTime += 1
                # print("\n\n\n\nqueue empty\n\n\n\n")
                self.master.after(1, self.process_queue)

class ThreadedLoadingBar(threading.Thread):
        # class to go with passage playing, -- updates frame whilst playing audio
        def __init__(self, page, queue1, cropBounds, video, success, image, passage):
            super().__init__()
            self.queue = queue1
            self.page = page
            self.cropBounds = cropBounds
            self.video = video
            self.success = success
            self.image = image
            self.passage = passage
        def run(self):
            y1, y2, x1, x2 = int(self.cropBounds["H"][0]), int(self.cropBounds["H"][1]), int(self.cropBounds["W"][0]), int(self.cropBounds["W"][1]) # Defining crop based on cropBounds
            numFrames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
            vidCapStep = (5000/numFrames)/3
            prev_frame_downscaled = 0
            timeIt.Start("video capture")
            for i in range(numFrames):
                self.success, self.image, prev_frame_downscaled = self.page.getFrames(i, self.video, y1, y2, x1, x2, self.success, self.image, prev_frame_downscaled)
                if i % 50 == 0: 
                    self.page.queue.put(vidCapStep)
                    print("put", vidCapStep, "- total val:", self.page.barVal)
            timeIt.Stop()
            self.page.parse(self.passage, y1, y2, x1, x2)

class FinalPg(tk.Frame):
    """
    The video has finished processing window
    """
    def __init__(self, parent, controller, debug):
        self.controller = controller
        self.debug = debug
        tk.Frame.__init__(self, parent)

        # Heading
        label = ttk.Label(self, text ="Video Processed", font = LARGEFONT)
        label.grid(row = 0, column = 1, padx = 10, pady = 10)

        def goNext():

            # Shuffle results, flags, and flagged frames into correct format.
            # basically stacking each of the indivdual arrays into their own set of arrays
            global R, F, FF, bF
            global tR, tF, tFF, tbF

            # Making sure each of the arrays is the same length for stacking
            if singlePassage:
                # only the passage array will have an array and len
                passage = controller.shared_data["Passage"].get()

                # rLength = len(rPrecursor[passage])
                timeIt.Start("Empty Passages")
                shape = len(rPrecursor[passage])
                cropList = self.controller.shared_data["crop"].get().split() # grabs crop listing from shared data
                cropBounds = {"H":[cropList[0],cropList[1]],
                              "W":[cropList[2],cropList[3]]}
                y1, y2, x1, x2 = int(cropBounds["H"][0]), int(cropBounds["H"][1]), int(cropBounds["W"][0]), int(cropBounds["W"][1]) # Defining crop based on cropBounds
                
                for i in range(17):
                    if i == passage: continue
                    # TODO: fix r into strings, and maybe f into int?
                    rPrecursor[i] = np.zeros((shape)) # R is array of int, so 0 is empty array 
            
                    fPrecursor[i] = np.zeros((shape)) # F is array of str, so "" is empty array

                    ffPrecursor[i] = np.zeros((shape, (y2-y1),(x2-x1),3)) # FF is array of arrays of arrays, so empty array

                    bfPrecursor[i] = np.zeros((len(bfPrecursor[passage]), 2)) # FF is array of arrays of arrays, so empty array
                timeIt.Stop()
                longest = passage
                longestBF = passage




                # # rLength = len(rPrecursor[passage])
                # timeIt.Start("Empty Passages")
                # Tshape = len(TrPrecursor[passage])                
                # for i in range(17):
                #     if i == passage: continue
                #     # TODO: fix r into strings, and maybe f into int?
                #     TrPrecursor[i] = np.zeros((Tshape)) # R is array of int, so 0 is empty array 
            
                #     TfPrecursor[i] = np.zeros((Tshape)) # F is array of str, so "" is empty array

                #     TffPrecursor[i] = np.zeros((Tshape, (y2-y1),(x2-x1),3)) # FF is array of arrays of arrays, so empty array

                #     TbfPrecursor[i] = np.zeros((len(TbfPrecursor[passage]), 2)) # FF is array of arrays of arrays, so empty array
                # timeIt.Stop()
                # Tlongest = Tshape
                # TlongestBF = len(TbfPrecursor[passage])

            else:
                cropList = self.controller.shared_data["crop"].get().split() # grabs crop listing from shared data
                cropBounds = {"H":[cropList[0],cropList[1]],
                              "W":[cropList[2],cropList[3]]}
                y1, y2, x1, x2 = int(cropBounds["H"][0]), int(cropBounds["H"][1]), int(cropBounds["W"][0]), int(cropBounds["W"][1]) # Defining crop based on cropBounds
    
                longest = 0
                for i in range(len(rPrecursor)):
                    if len(rPrecursor[i])>len(rPrecursor[longest]): longest = i

                longestBF = 0
                for i in range(len(bfPrecursor)):
                    if len(bfPrecursor[i])>len(bfPrecursor[longestBF]): longestBF = i

                for i in range(17):
                    if i == longest: continue
                    else:
                        shape = len(rPrecursor[longest])-len(rPrecursor[i]) # all same length
                        FFshape = (shape,(y2-y1),(x2-x1),3)
                        # empty is the technically fastest, but doesn't leave consistent values so we use full and zeros
                        # zeros is considerably quicker than full, so it's used for the int ones
                        # TODO: fix R into strings, maybe f into int?
                        rAdd = np.full((shape), "") # R is array of str, so "" is empty array
                        rPrecursor[i] = np.concatenate((rPrecursor[i],rAdd), axis=None)

                        fAdd = np.full((shape), "") # F is array of str, so "" is empty array
                        fPrecursor[i] = np.concatenate((fPrecursor[i],fAdd), axis=None)

                    # if i == longestBF: continue
                    # else:
                    #     bShape = (17, longestBF, 2)
                    #      # FF is array of arrays of arrays, so empty array
                    #     bfPrecursor[i] = np.concatenate((bfPrecursor[i],bfAdd), axis=0)
                    True

                




            shape = (17,len(rPrecursor[longest]),(y2-y1),(x2-x1),3)
            bShape = (17, len(bfPrecursor[longestBF]), 2)
            timeIt.Start("Stacking Arrays")
            R = np.vstack([rPrecursor[i] for i in range(17)])
            F = np.vstack([fPrecursor[i] for i in range(17)])
            FF = np.empty(shape=shape, dtype="uint8")
            for i in range(17):
                if (len(ffPrecursor[longest])-len(ffPrecursor[i])) > 0:
                    FFshape = ((len(ffPrecursor[longest])-len(ffPrecursor[i])),(y2-y1),(x2-x1),3)
                    ffAdd = np.zeros((FFshape), dtype="uint8") # FF is array of arrays of arrays, so empty array
                    _e1 = np.concatenate([ffPrecursor[i],ffAdd], axis=0)
                    FF[i,:,:,:,:] = np.concatenate((ffPrecursor[i],ffAdd), axis=0)
                else: FF[i,:,:,:,:] = ffPrecursor[i]
            bF = np.empty(shape=bShape, dtype="uint16")
            for i in range(17):
                if i != longestBF:
                    bfShape = (len(bfPrecursor[longestBF])-len(bfPrecursor[i]),2)
                    a = bfPrecursor[i] # bf clear here
                    bfAdd = np.zeros((bfShape), dtype="uint16")
                    # print(bfPrecursor[i])
                    bF[i,:,:] = np.concatenate((bfPrecursor[i],bfAdd), axis=0)#! this kills it 
                else:
                    bF[i,:,:] = bfPrecursor[i]
            timeIt.Stop()
            x = bF[0] 
            True

            for i, pas in enumerate(bF):
                bF[i] = bF[i][bF[i][:, 1].argsort()]
            y = bF[0]
            True




            # if False:
            #     Tshape = (17,Tlongest,(y2-y1),(x2-x1),3)
            #     TbShape = (17, TlongestBF, 2)
            #     timeIt.Start("Stacking Arrays")
            #     tR = np.vstack([TrPrecursor[i] for i in range(17)])
            #     tF = np.vstack([TfPrecursor[i] for i in range(17)])
            #     tFF = np.empty(shape=shape, dtype="uint8")
            #     for i in range(17):
            #         tFF[i,:] = TffPrecursor[i]
            #     tbF = np.empty(shape=TbShape, dtype="uint16")
            #     for i in range(17):
            #         a = TbfPrecursor[i] # bf clear here
            #         # print(bfPrecursor[i])
            #         tbF[i,:,:] = TbfPrecursor[i]#! this kills it 
            #     timeIt.Stop()
            #     x = tbF[0] 
            #     True

            #     for i, pas in enumerate(tbF):
            #         tbF[i] = tbF[i][tbF[i][:, 1].argsort()]
            #     y = tbF[0]
                # True

            # df = pd.DataFrame([[y, x] for x, y in enumerate(R[0]) if "FLAG" not in y],columns=['decibels','frames'])

            # if False:
                
            #     rrrr = np.stack((np.array(R[passage]),np.array(tR[passage])))
            #     drrr = pd.DataFrame((rrrr), index=["Threaded","Unthreaded"])

            #     x, y = zip(*[[x, int(y)] for x, y in enumerate(R[passage]) if "FLAG" not in y])
            #     tx, ty = zip(*[[x, int(y)] for x, y in enumerate(tR[passage]) if "FLAG" not in y])
            #     # print(y)
            #     # print(ty)

                
            #     fig, main_ax = plt.subplots()
            #     line1 = main_ax.plot(x, y, label="Threaded")
            #     line2 = main_ax.plot(tx, ty, label="Unthreaded")
            #     # main_ax.set_xlim(0, max(x))
            #     ymax, ymin = int(max(y))+3, int(min(y))-3
            #     print("Max:",int(max(y))+3,"- Min:",int(min(y))-3)
            #     # main_ax.set_ylim(ymax, ymin)
            #     # main_ax.yaxis.set_inverted(True)
            #     main_ax.set_xlabel("Frames")
            #     main_ax.set_ylabel("Decibels")
            #     main_ax.set_title('Before Flags Page')
            #     main_ax.legend()
            #     plt.show() 
            #     print(R[0,0])
            #     if int(R[passage, 0]) < 55:
            #         print("R out of order?")
            #         True
            True


            # class Particle(tables.IsDescription):
            #     name      = tables.StringCol(16)   # 16-character String
            #     idnumber  = tables.Int64Col()      # Signed 64-bit integer
            #     ADCcount  = tables.UInt16Col()     # Unsigned short integer
            #     TDCcount  = tables.UInt8Col()      # unsigned byte
            #     grid_i    = tables.Int32Col()      # 32-bit integer
            #     grid_j    = tables.Int32Col()      # 32-bit integer
            #     pressure  = tables.Float32Col()    # float  (single-precision)
            #     energy    = tables.Float64Col()    # double (double-precision)

            # h5file = tables.open_file("storedData.h5", mode="w", title="storedData")
            # group = h5file.create_group("/", 'detector', 'Detector information')
            # table = h5file.create_table(group, 'readout', Particle, "Readout example")
            # particle = table.row
            # for i in range(10):
            #     particle['name']  = f'Particle: {i:6d}'
            #     particle['TDCcount'] = i % 256
            #     particle['ADCcount'] = (i * 256) % (1 << 16)
            #     particle['grid_i'] = i
            #     particle['grid_j'] = 10 - i
            #     particle['pressure'] = float(i*i)
            #     particle['energy'] = float(particle['pressure'] ** 4)
            #     particle['idnumber'] = i * (2 ** 34)
            # particle.append()
            # table.flush()



            True

            # h5file.close()

            controller.show_frame(FlagsPg)

        # Move to viewing flags
        button1 = ttk.Button(self, text ="View Flagged Frames", command = goNext)
        button1.grid(row = 1, column = 1, padx = 10, pady = 10)

class FlagsPg(tk.Frame):
    """
    The flags correction window.
    This is kind of a nightmare, I apologise.

    TODO: finish commenting this frame after it's been fixed.
    """
    def __init__(self, parent, controller, debug):
        self.controller = controller
        self.debug = debug
        tk.Frame.__init__(self, parent)

        # Heading
        label = ttk.Label(self, text ="Flagged Images", font = LARGEFONT)
        label.grid(row = 0, column = 3, padx = 10, pady = 10)

        # Setting important variables for this frame (window).
        self.flagNum, self.remaining, self.flagLoc, self.iterations = 0, 0, 0, 0
        self.flagVal=tk.StringVar()
        self.passage, self.passageNum = 0, 0
        self.finiButton = False
        self.initialised = False

        # Declaring GUI elements 
        self.nextButton = ttk.Button(self, text ="View Next Flag", command = self.nextFlag) # Next Flag Button
        self.finishButton = ttk.Button(self, text ="Finish", command = lambda : self.submit(list(bF[self.passage].values()))) # On last flag, next flag button goes away and this finish button appears
        self.remainingLabel = ttk.Label(self, text =f"Remaining Flagged Images: {self.remaining}") # Remaining images that need input
        self.flagEntry = tk.Entry(self,textvariable = self.flagVal) # The box to type into
        self.guessLabel = ttk.Label(self, text =f"Placeholder") # The guess of what the digits are. Currently a placeholder.
        self.backB = ttk.Button(self, text ="Back", command = self.back) # Back button
        self.space = ttk.Label(self, text =f"                    ") # Spacer
        self.oneUp = ttk.Button(self, text=">") # Flag is one more than guess button
        self.oneDown = ttk.Button(self, text="<") # Flag is one less than guess button
        self.lastFlag = "None"
        
        # Start Button
        self.startB = ttk.Button(self, text="Start Viewing Flagged Frames", command = self.flaggedFrames)
        self.startB.grid(row=5, column=3, padx=10, pady=10)

    def nextKey(self,event):
        """On enter, move to next image.

        :param event: `<Return>` button press
        :type event: tk event
        """
        self.nextFlag()

    def guessUp(self, event): 
        """
        Sets flag value to be one higher than guess, and updates the entry box.
        """
        self.flagVal.set(str(int(self.guess(self.flagIndex))+1))
        self.flagEntry.config(textvariable=self.flagVal)
    def guessDown(self, event): 
        """
        Sets flag value to be one lower than guess, and updates the entry box.
        """
        self.flagVal.set(str(int(self.guess(self.flagIndex))-1))
        self.flagEntry.config(textvariable=self.flagVal)
    def resetKey(self,event):
        """On up arrow key press, set the flag value to be the guess. This is for quickly resetting after hitting one of the other arrow keys.

        :param event: `<Up>` button press
        :type event: tk event
        """
        self.flagVal.set(self.guess(self.flagIndex))
        self.flagEntry.config(textvariable=self.flagVal)

    def guess(self, ind: int) -> str:
        """Guessing based on previous results

        :param ind: Current flagged index
        :type ind: int
        :return: The guess based on previous results
        :rtype: str
        """
        for result in np.flipud(R[self.passage][:ind]): # enumerate from the previous index, backwards
            if "FLAG" not in result: return result

    def submit(self):
        """On hitting next flag, or submit, save data.

        :param bFv: The values in bF (batched flags dictionary, but just the list of values)
        :type bFv: list
        """
        global F, R
        val = self.flagVal.get() # grabbing flag value

        if val == "": # if the value was not inputted, the guess was correct
            val = self.guess(self.flagIndex)
        
        # TODO: once fix, comment this

        """dealing with batched flags
        # self.iterations = 0
        # for i,x in enumerate(bFv[self.flagNum]):
        #     if self.flagLoc+i >= len(F[self.passage]):
        #         print("something wrong?")
        #         break
        #     # if self.flagLoc+i == len(F[self.passage]): break # might delete
        #     print("(loc in flags + i) flagloc+i", self.flagLoc+i,"| i", i, "| flag val",F[self.passage][self.flagLoc+i][1], "| num flags",len(F[self.passage]), "| num batched flags",len(bFv), "| location in batched flags",self.flagNum)
        #     R[self.passage][F[self.passage][self.flagLoc+i][0]] = val
        #     self.iterations = i
        # print(self.flagLoc)
        # self.flagLoc += self.iterations + 1
        # self.flagNum += 1
        # print(self.flagLoc)"""

        # bF layout = FLAG VALUE, repeats in a row, original index

        R[self.passage][self.flagIndex] = val
        print("set", self.flagIndex, "as", val)
        
        a1 = bF[self.passage]
        a2 = bF[self.passage, self.bFcounter]
        a3 = bF[self.passage, self.bFcounter, 0]

        rep, ind = bF[self.passage, self.bFcounter, 0], bF[self.passage, self.bFcounter, 1]
        for i in range(1, rep):
            R[self.passage][ind+i] = val
            print("batched", ind+i, "as", val)



        if "FLAG" in R[self.passage, self.flagIndex]:
            print("uhoh didnt remove")
            True

        for ii, res in enumerate(np.flipud(R[self.passage][:self.flagIndex])):
            if "FLAG" in res:
                skipped = self.flagIndex-ii-1
                print("skipped somehow???")
                True
    
        
        self.flagIndex = self.findFlag(bF[self.passage],self.bFcounter)
        self.bFcounter += 1
        self.remaining -= 1

        
        if self.remaining < 3:
            True
        self.flagVal.set("") # resetting flag value

    def nextFlag(self):
        """
        This displays all flags except for the first one, and stores data.
        """
        global F, R, FF, bF

        self.submit() # save previous flag value
        if self.remaining < 0: 
            print(self.passageNum)
            self.controller.unbind("<Return>")
            self.controller.unbind("<Left>")
            self.controller.unbind("<Right>")
            self.controller.show_frame(DownPg)
            return

        # Update Labels
        self.remainingLabel.config(text =f"Remaining Flagged Images: {self.remaining}")
        self.guessLabel.config(text =f"Is this {self.guess(self.flagIndex)}? (Leave blank, if yes)")

        # Update Image
        image = CVtoPIL(FF[self.passage,self.flagIndex])
        imageLabel = ImageTk.PhotoImage(image)
        self.label1.config(image=imageLabel)
        self.label1.image = imageLabel 
        self.label1.grid(row = 2, column = 3, padx = 10, pady = 10)
        # ! might need this? unsure, seems like doing it twice imo
        
        # Add back button
        if not self.initialised:
            self.backB.grid(row = 4, column = 1, padx = 10, pady = 10)
            self.space.grid_forget()
            self.initialised = True

    def findFlag(self, flags, currIndex):
        # timeIt.Start("flag 1")
        # flags1 = [i for i in flags if 0 not in i[0]]
        # timeIt.Stop()
        # timeIt.Start("flag 2")
        # mask = np.isin(flags, [0])
        # maskedFlags = flags[~mask]
        # _e = len(maskedFlags)
        # clearFlags = np.reshape(maskedFlags, (int(len(maskedFlags)/2), 2))
        # timeIt.Stop()


        if currIndex > len(flags)-2 or len(flags) == 0: return self.finalFlag()
        else:
            self.movedNext = False
            if currIndex+1 == self.lastFlagIndex and (self.passageNum == 7 if not singlePassage else True): 
                self.finButton = True
                self.nextButton.grid_forget()
                self.finishButton.grid(row = 4, column = 5, padx = 10, pady = 10)
            if currIndex >= len(flags)-2:
                True 
            print("debug returning index")
            return flags[currIndex+1][1]


    def finalFlag(self) -> (int | None):
        print("debug final flag started")
        # no next flag found, checking for future flags
        if singlePassage or self.passageNum == 16:
            if self.remaining != 0: print("something went wrong, remaining != 0 when done with passage/s")

            # after submitting final flag, show download page
            print("reached length")
            self.controller.unbind("<Return>")
            self.controller.unbind("<Left>")
            self.controller.unbind("<Right>")
            return "none"

        else: # next passage
            print("\n\nnew passage\n\n")
            skipped = False
            for i, res in enumerate(R[self.passage]):
                if "FLAG" in res:
                    print("skipped:", i)
                    skipped = True
            if skipped: print(R[self.passage])
            

            self.passageNum += 1
            self.passage = self.passages[self.passageNum]
            self.movedNext = True
            vals, counts = np.unique(bF[self.passage], axis=0, return_counts=True)
            if np.count_nonzero(vals[0]) > 0: 
                self.bFcounter = -1
                True
            else: self.bFcounter = counts[0]-1
            return self.findFlag(bF[self.passage],self.bFcounter) # back to square 1, for new passage
        
    def flaggedFrames(self):
        """
        The first page of flagged images. (This is the first flagged image.) 
        This primarily sets things up for the next flags.
        """
        self.startB.grid_forget() # remove start button
        
        # add necessary buttons & boxes
        self.nextButton.grid(row = 4, column = 5, padx = 10, pady = 10) 
        self.flagEntry.grid(row = 4, column = 3, padx = 10, pady = 10)
        self.space.grid(row = 4, column=1, padx=10,pady=10)

        self.oneUp.grid(row = 4, column=4, padx=10,pady=10)
        self.oneDown.grid(row = 4, column=2, padx=10,pady=10)
        
        # set up enter
        self.controller.bind("<Return>", self.nextKey)
        # setting up arrow keys
        self.controller.bind("<Left>", self.guessDown)
        self.controller.bind("<Right>", self.guessUp)
        self.controller.bind("<Up>", self.resetKey)

        global F, R, FF, bF

        # Determine Passage
        if singlePassage: self.passage = self.controller.shared_data["Passage"].get()
        else:
            rOrder = self.controller.shared_data["PassageOrder"].get()
            self.passages = listenOrders[Orders[rOrder, 0]]
            self.passage = self.passages[self.passageNum]


        if singlePassage: self.flags = len(bF[self.passage])
        else: 
            _w1 = [o for o in bF]
            _w2 = np.ravel([o for o in bF])
            _w3 = np.ravel([o for o in bF if 0 not in o])
            flagsTotal = []
            for eo in bF:
                flagsTotal.extend(list(np.ravel([o for e, o in eo if e != 0])))
            self.flags = len(flagsTotal)
            True
        
        p = np.flipud(bF[self.passage])
        for i, item in enumerate(np.flipud(bF[self.passage])):
            if item[1] == bF[self.passage][0][1]:  # this theoretically shouldn't occur
                # ! if needed, move back a passage, and retest from last index
                print("something went wrong: went back to index of 0")
            self.lastFlagIndex = bF[self.passage][-1][1]
            break

        vals, counts = np.unique(bF[self.passage], axis=0, return_counts=True)


        True
        if np.count_nonzero(vals[0]) > 0: 
            _e = np.count_nonzero(vals[0]) 
            _e1 = np.count_nonzero(vals)
            print("longest:", self.passage)
            self.passFlags, self.flagIndex = len(bF[self.passage]), self.findFlag(bF[self.passage],-1)
            self.bFcounter = 0
            True
        else:
            self.passFlags, self.flagIndex = len(bF[self.passage]), self.findFlag(bF[self.passage],counts[0]-1)
            self.bFcounter = counts[0]

        """
        :var passFlags: Number of flags in specific passage
        :type passFlags: int

        :var flagIndex: Index of the current flag in specific passage
        :type flagLoc: int

        :var flags: Number of flags in total
        :type flags: int
        """
        if not singlePassage: self.totalBF = len(np.ravel(bF))
        else: self.totalBF = len(bF[self.passage])
        if self.totalBF != self.flags: print("batched flags different number than flags:", self.totalBF, self.flags)
        self.remaining = self.flags - 1 # -1 because actively displaying an image at the time
        self.remainingLabel.config(text =f"Remaining Flagged Images: {self.remaining}")
        self.remainingLabel.grid(row = 1, column = 3, padx = 10, pady = 10)

        # display image
        image = CVtoPIL(FF[self.passage,self.flagIndex])
        imageLabel = ImageTk.PhotoImage(image)
        self.label1 = tk.Label(self, image=imageLabel)
        self.label1.image = imageLabel
        self.label1.grid(row = 2, column = 3, padx = 10, pady = 10)
        

        self.guessLabel.config(text =f"Is this {self.guess(self.flagIndex)}? (Leave blank, if yes)") # guess
        self.guessLabel.grid(row = 3, column = 3, padx = 10, pady = 10)

        self.oneUp.config(command= self.guessUp(""))
        self.oneDown.config(command= self.guessDown(""))
        self.flagVal.set("")
        self.flagEntry.config(textvariable=self.flagVal)

    def BACKfindFlag(self, flags, currIndex):
        currIndex = self.BACKbatch(currIndex)
        True
        return currIndex
        # p = np.flipud(flags[:currIndex])
        # True
        # for i, item in enumerate(np.flipud(flags[:currIndex])):
        #     if item == 0:  # this theoretically shouldn't occur
        #         # ! if needed, move back a passage, and retest from last index
        #         print("something went wrong: went back to index of 0")
        #     if "FLAG" in item: 
        #         e = len(flags[:currIndex])-i-1
        #         return e # because reversed
            
    def BACKbatch(self, currIndex):
        global bF
        print(currIndex)
        # bF layout = FLAG VALUE, repeats in a row, original index
        self.bFcounter -= 1

        a1 = bF[self.passage]
        a2 = bF[self.passage, self.bFcounter]
        a3 = bF[self.passage, self.bFcounter, 0]
        return bF[self.passage, self.bFcounter, 1]

    def back(self): # confident that this is broken currently -- something something rewrote/fixed bad code
        """
        When back button is pressed, actually go back. (And make sure nothing breaks)
        """
        global F, R, FF

        # if it's the last flag, we don't want the finish button to stick around
        if self.finiButton:
            self.finishButton.grid_forget()
            self.nextButton.grid(row = 4, column = 5, padx = 10, pady = 10)

        # if moved next, move back a passage
        if self.movedNext:
            self.passageNum -= 1
            self.passage = self.passages[self.passageNum]
            self.movedNext = False
            self.flagIndex = len(F[self.passage]-1)
            True

        # If it's the first flag, needs a different view.
        if self.remaining == (self.flags-2):
            self.backB.grid_forget() # removing back button (can't go back from first)
            self.initialised = False

            # redisplaying previous 
            self.remaining += 1
            self.remainingLabel.config(text =f"Remaining Flagged Images: {self.remaining}")

            self.flagIndex = self.BACKfindFlag(F[self.passage], self.flagIndex)
            image = CVtoPIL(FF[self.passage,self.flagIndex])
            imageLabel = ImageTk.PhotoImage(image)
            self.label1 = tk.Label(self, image=imageLabel)
            self.label1.image = imageLabel
            self.label1.grid(row = 2, column = 3, padx = 10, pady = 10)

            self.guessLabel.config(text =f"Is this {self.guess(self.flagIndex)}? (Leave blank, if yes)")
        else:
            # redisplaying previous 
            self.remaining += 1
            self.remainingLabel.config(text =f"Remaining Flagged Images: {self.remaining}")

            self.flagIndex = self.BACKfindFlag(F[self.passage], self.flagIndex)

            image = CVtoPIL(FF[self.passage,self.flagIndex])
            imageLabel = ImageTk.PhotoImage(image)
            self.label1 = tk.Label(self, image=imageLabel)
            self.label1.image = imageLabel
            self.label1.grid(row = 2, column = 3, padx = 10, pady = 10)

            self.guessLabel.config(text =f"Is this {self.guess(self.flagIndex)}? (Leave blank, if yes)")

class DownPg(tk.Frame):
    """ 
    The download window
    """
    def __init__(self, parent, controller, debug):
        self.controller = controller
        self.debug = debug
        tk.Frame.__init__(self, parent)

        # Heading
        label = ttk.Label(self, text ="Download Data", font = LARGEFONT)
        label.grid(row = 0, column = 1, padx = 10, pady = 10)
        self.controller.shared_data["ID"].set("0") # TODO: i think not needed

        def download():
            """
            When download button is pressed, download data.
            
            Filename, when set: `results_P{ID}`
            Filename, when singlePassage: `results_P{ID}_Pass{Passage}`
            To be included in file: ID, Passage, Passage Difficulty, Time/Frame, Decibels
            """
            global R
            # Setting ID and Passage order
            ID, PassageOrder = self.controller.shared_data["ID"].get(), self.controller.shared_data["PassageOrder"].get() # grabbing stored values

            # Determining filename
            PATH = os.path.join(base_dir, f'Results/results_P{ID}' if not singlePassage else f"Results/resultsP{ID}_Pass{self.controller.shared_data["Passage"].get()}")
            with open(PATH, 'w') as f: # writing file
                write = csv.writer(f) # storing file as CSV file
                write.writerow(["ID","RO","Passage","Difficulty","Frame", "Decibels"]) # Row titles
                for passage, result in enumerate(R): # for each passage listed, keep going
                    a = self.controller.shared_data["Passage"].get()
                    if singlePassage and self.controller.shared_data["Passage"].get() != passage: continue
                    PassageDiff = PDiffs[passage] # grabbing passage difficulty based on passage
                    for i, r in enumerate(result): print(r, end=" ")
                    res = [[ID,(PassageOrder+1 if not singlePassage else "S"),passage,PassageDiff,frame, decibels] for frame, decibels in enumerate(result) if decibels != ""]
                    write.writerows(res)
                    if singlePassage:
                        x, y = zip(*[[x, int(y)] for x, y in enumerate(result)])
                        fig, main_ax = plt.subplots()
                        main_ax.plot(x, y)
                        print("Max:",int(max(y))+3,"- Min:",int(min(y))-3)
                        main_ax.set_xlabel("Frames")
                        main_ax.set_ylabel("Decibels")
                        main_ax.set_title('Before Flags Page')
                        plt.show() 
                        if True:
                            True
            

        # Download Button
        downB = ttk.Button(self, text="Download", command = download)
        downB.grid(row=5, column=1, padx=10, pady=10)
        
class cropPg(tk.Frame):
    """
    The Cropping Page
    """
    def __init__(self, parent, controller, debug):
        self.controller = controller
        self.debug = debug
        tk.Frame.__init__(self, parent)
        
        # Heading 
        label = ttk.Label(self, text ="Crop Video", font = LARGEFONT)
        label.grid(row = 0, column = 2, padx = 10, pady = 10,columnspan=2)
        
        # Defining Canvas and x/y points
        self.x = self.y = 0
        self.canvas = tk.Canvas(self, cursor="cross")

        self.frames, self.nexts = [], 0 # List of frames and number of times hit next frame

        # Defining crop bounds dictionary
        self.crop = {"H":[0,0], # height, lower to higher bounds
                     "W":[0,0]} # width, lower to higher bounds

        def nextImage():
            """
            On next frame button press, display next image.
            """
            self.nexts += 1 # Set next up one for next image
            
            success,image = self.video.read() # reading next frame

            x1, x2, y1, y2 = 892, 998, 242, 308
            curr_frame = image[y1:y2,x1:x2]
            # curr_frame_downscaled = cv2.resize(curr_frame, (400, 400), cv2.INTER_AREA)
            grey = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY) # greyscale
            blur = cv2.GaussianBlur(grey, (5,5), 0) # blurred
            thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] # this makes the contours pop out
            invert = 255 - thresh

            diff_index = np.sum(np.abs(invert - self.prev))
            
            
            print(f"Changed: {diff_index>40000}. Diff Index: {diff_index}.")

            self.prev = invert


            # image = invert

            # Convert next image
            image = CVtoPIL(image)
            width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))/2
            wpercent = (float(width) / float(image.size[0]))
            hsize = int((float(image.size[1]) * float(wpercent)))
            image = image.resize((int(width), int(hsize)), Image.Resampling.LANCZOS)

            # Display next image
            self.imageLabel = ImageTk.PhotoImage(image)
            self.canvas.itemconfig(self.image_on_canvas, image = self.imageLabel)

        def crop():
            """
            Actually cropping the image

            Display image, and allow for box to be made where wanted.
            """
            # Remove unneeded buttons, and add necessary buttons
            cropB.grid_forget()
            nextFb.grid(row=3, column=3, padx=1, pady=1)
            self.resetCrop.grid(row=3, column=2, padx=1,pady=1)

            # Grabbing first frame
            global singlePassage
            videoFile = self.controller.shared_data["video"].get() if singlePassage else files[0]
            self.video = cv2.VideoCapture(videoFile) # grabbing video
            width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))/2
            height = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))/2
            success,image = self.video.read() # reading video into image

            x1, x2, y1, y2 = 892, 998, 242, 308
            curr_frame = image[y1:y2,x1:x2]
            # curr_frame_downscaled = cv2.resize(curr_frame, (400, 400), cv2.INTER_AREA)
            grey = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY) # greyscale
            blur = cv2.GaussianBlur(grey, (5,5), 0) # blurred
            thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] # this makes the contours pop out
            invert = 255 - thresh
            self.prev = invert


            # image = invert



            # Convert image to Pillow format for displaying image on GUI
            image = CVtoPIL(image)
            wpercent = (float(width) / float(image.size[0]))
            hsize = int((float(image.size[1]) * float(wpercent)))
            image = image.resize((int(width), int(hsize)), Image.Resampling.LANCZOS)
            
            # Setting up image labelling
            self.imageLabel = ImageTk.PhotoImage(image)
            
            # Displaying image on canvas
            self.image_on_canvas = self.canvas.create_image(0,0,anchor="nw",image=self.imageLabel,state=tk.NORMAL)
            self.canvas.config(width=width,height=height/2) # yeah this is weird, but for some reason the height was double and messing with everything. 
            self.canvas.grid(row=2,column=2,padx = 1,pady = 1, columnspan=2)
                
            # Binding mouse button and movement to display the rectangle
            self.canvas.bind("<ButtonPress-1>", self.on_button_press)
            self.canvas.bind("<B1-Motion>", self.on_move_press)
            self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        def resetCrop():
            self.canvas.delete(self.rect)
            self.rect = None
            self.start_x = None
            self.start_y = None
            self.nextButton.config(style="DisabledB.TButton",state = 'disabled')
            self.resetCrop.config(style="DisabledB.TButton",state = 'disabled')
    
        def nextF():
            """
            Save crop and move to next frame
            """
            if self.video != []: self.video.release() # release video from memory if not already gone
            self.crop["W"][0], self.crop["H"][0], self.crop["W"][1], self.crop["H"][1] = self.canvas.coords(self.rect) # Set crop bounds to the coordinates of rectangle
            self.controller.shared_data["crop"].set(f"{int(self.crop["H"][0])*2} {int(self.crop["H"][1])*2} {int(self.crop["W"][0])*2} {int(self.crop["W"][1])*2}") # save crop bounds in shared data
            
            # Move to next frame depending on singlePassage
            global singlePassage
            self.controller.show_frame(WorkingSetPg if not singlePassage else WorkingPg)

        # Crop Image Button
        cropB = ttk.Button(self, text="Start Cropping", command = crop)
        cropB.grid(row=2, column=2, padx=10, pady=10,columnspan=2)
        
        def back():
            global files
            controller.show_frame(UploadPg)
            if not singlePassage: files = []
            else: self.controller.shared_data["video"].set("") # stores filename if singlePassage

        # Back Button
        backB = ttk.Button(self, text ="Back", command = back)
        backB.grid(row = 3, column = 1, padx = 10, pady = 10)

        # Next button
        self.nextButton = ttk.Button(self, text ="Next", style="DisabledB.TButton", command=nextF, state = 'disabled')
        self.nextButton.grid(row = 3, column = 4, padx = 10, pady = 10)
        
        self.video = [] # sets base variable

        # Next frame button
        nextFb = ttk.Button(self, text="New Frame", command = nextImage)
        self.resetCrop = ttk.Button(self, text="Reset Crop", style="DisabledB.TButton", command = resetCrop, state='disabled')


        # Defines base rectangle
        self.rect = None
        self.start_x = None
        self.start_y = None

    
    def get_mouse_posn(self, event):
        """Gets current mouse position
        
        TODO: I think never used

        :param event: idk
        :type event: tk event
        """
        self.topx, self.topy = event.x, event.y

    def on_button_press(self, event):
        """On mouse button press, start placing the rectange

        :param event: `<ButtonPress-1>`: mouse button 1 clicked
        :type event: tk event
        """
        # save mouse drag start position
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)

        # create rectangle if it doesn't exist yet
        if not self.rect:
            self.rect = self.canvas.create_rectangle(self.x, self.y, 1, 1, outline='red')

    def on_move_press(self, event):
        """On mouse movement, move the bounds of the rectangle with the mouse.

        :param event: `<B1-Motion>`: mouse movement
        :type event: tk event
        """
        # sets current x & y
        curX = self.canvas.canvasx(event.x)
        curY = self.canvas.canvasy(event.y)
        # expand rectangle as you drag the mouse
        self.canvas.coords(self.rect, self.start_x, self.start_y, curX, curY)

    def on_button_release(self, event):
        """On mouse button release, allow next button to be pressed.

        :param event: `<ButtonRelease-1>`: mouse button 1 released
        :type event: tk event
        """
        self.nextButton.config(style="BlueB.TButton",state = 'enabled')
        self.resetCrop.config(style="BlueB.TButton",state = 'enabled')



# Driver Code
app = tkinterApp()
app.mainloop()