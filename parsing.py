"""

Author: Griffin Dugan
Brandeis Memory & Cognition (Wingfield) Lab
EVE1 Parsing Program

Description: Required file of parsing functions for gui.py to run. This file contains the main parsing functions that are used to parse the images and find the results.
"""

from math import e
from OCR import findText, flagWeird, finalPlausibilityCheck, sortFlags, timeIt, fixFlags, mTest, flag

import threading
import numpy as np
from collections import defaultdict as dd
from queue import Queue
import itertools as it
import functools as ft
import time

import pandas as pd


def l2np(lst:list) -> np.ndarray:
    # ? dont think this is nec
    a = []
    for x in lst:
        a.append(x)
    a = np.array(a)

def threadedParse(oldFrames:list, crop:tuple, page:object) -> tuple[list, list, list]:
    # if frames == []: return [], [], {}
    def worker(index, re,fl,ffl):
        frameList = q.get()
        re[index], fl[index], ffl[index] = parse(frameList, crop, page, page.lenFrames)
        print("thread",index+1,"is now done")
        q.task_done()

    # * Option 1: Not caring about how many true frames go to each thread
    #   - Positives: less memory intensive (i think)
    #   - Cons: threads finish at wildly different rates, and might slow things down significantly
    # * Option 2: Caring about the number and accurately splitting.


    shape = oldFrames.shape
    totalFrames = len(oldFrames)
    test = [(y, i) for i, y in enumerate(oldFrames) if y.any() != 0]
    test2 = oldFrames[0].any() != 0

    # * This is the first step in the process, where the frames are cleaned and the length of the frames is updated.
    frames, frameCount = zip(*[(y, i) for i, y in enumerate(oldFrames) if y.any() != 0]) # removing empty frames
    frames, frameCount = np.asarray(frames), np.asarray(frameCount) # converting to numpy arrays
    page.lenFrames = len(frames) # updating the length of frames

    # * This is the second step in the process, where the frames are split into 4 parts, and then each part is sent to a thread to be parsed.
    timeIt.Start("Split Frames")
    splitFrames = np.array_split(frames, 4) # splitting the frames into 4 parts
    splitFramecount = np.array_split(frameCount, 4) # ? idk what this is 
    timeIt.Stop()

    q = Queue() # creating a queue for the threads
    t, tr, tf, tff = np.empty((4),dtype=np.object_), np.empty([4],dtype=np.ndarray), np.empty([4],dtype=np.ndarray), np.empty([4],dtype=np.ndarray) # creating arrays for the threads, results, flags, and flagged frames
    # t = threads
    # r = results
    # f = flags
    # np.empty([4],dtype=np.str_)
    # (splitLength+remaining)

    # r, f, ff = np.empty((1),dtype=np.ndarray), np.empty((1),dtype=np.ndarray), np.empty((1),dtype=np.ndarray)

    # * This is the third step in the process, where the frames are parsed by the threads.
    timeIt.Start("Parsing")
    # f[0], f[1], f[2], f[3] = map(list, splitFrames)
    # for i in range(4):
    numThread = 4 # has to correlate to splitFrames splits
    for i in range(numThread): # for each thread
        # print(splitFramecount[i][0])
        q.put(splitFrames[i]) # put the frames into the queue
        t[i] = threading.Thread(target=worker, args=(i, tr,tf,tff)) # create the thread
    # q.put(frames)
    # t[4] = threading.Thread(target=worker, args=(0, r,f,ff))

    for i in range(numThread): # for each thread
        t[i].start() # start the thread
        time.sleep(0.01) # sleep for a bit to allow for the individual threads to start
    # t[4].start()

    q.join() # join the queue
    
    for i in range(numThread): # for each thread
        t[i].join() # join the threads
        time.sleep(0.01) # sleep for a bit to allow for the individual threads to join
        print(f"Thread {i+1} joined")
    # t[4].join()
    # print(f"Thread 5 joined")
    timeIt.Stop()

    # print("Flags Correct:", round(page.barVal/page.p,10) == round((100/17),10))

    _s1 = (100/17)
    _s2 = (100/17)/3
    _s3 = 2*(100/17)/3
    _s4 = ((100/17)/3)/page.lenFrames
    _s5 = (2*(100/17)/3)/page.lenFrames


    # * This is the fourth step in the process, where the frames are concatenated and the results are updated.
    frameshape = (totalFrames,) + crop # the shape of the frames
    # rPrime = np.empty((totalFrames), dtype="U15")
    # fPrime = np.empty((totalFrames), dtype="U15")
    # ffPrime = np.empty((frameshape), dtype="uint8")

    TrPrime = np.empty((totalFrames), dtype="U15")
    TfPrime = np.empty((totalFrames), dtype="U15")
    TffPrime = np.empty((frameshape), dtype="uint8")

    # TODO: check 1st and last index of each array for plausibility (excluding literal first and last of master array)

    
    timeIt.Start("Concatenating")
    # threaded
    tr = np.concatenate((tr[0],tr[1],tr[2],tr[3]), axis=None)
    tf = np.concatenate((tf[0],tf[1],tf[2],tf[3]), axis=None)
    tff = np.vstack((tff[0],tff[1],tff[2],tff[3]))

    # # unthreaded
    # R = np.array(r[0])
    # F = np.array(f[0])
    # fF = np.array(ff[0])
    # timeIt.Stop()

    # if int(R[0]) < 55:
    #     print("R out of order")
    #     True

    if "FLAG" in tr[0]:
        print("flag in 1st R") 
        True # TODO first flag error go again

    elif int(tr[0]) < 55:
        print("TR out of order")
        True

    # np.put(rPrime, frameCount, R)
    # np.put(fPrime, frameCount, F)
    # for i, count in enumerate(frameCount):
    #     ffPrime[count,:] = fF[i]

    np.put(TrPrime, frameCount, tr)
    np.put(TfPrime, frameCount, tf)
    for i, count in enumerate(frameCount):
        TffPrime[count,:] = tff[i]

    # frameChanged = -1
    # for index, count in enumerate(rPrime):
    #     if count != "": 
    #         frameChanged +=1
    #         continue
    #     if index > frameCount[frameChanged]:
    #         rPrime[index] = R[frameChanged]

    frameChanged = -1
    for index, count in enumerate(TrPrime):
        if count != "": 
            frameChanged +=1
            continue
        if index > frameCount[frameChanged]:
            TrPrime[index] = tr[frameChanged]

    # batching flags
    # * This is the fifth step in the process, where the flags are batched.
    # bF = batchFlags(rPrime)
    TbF = batchFlags(TrPrime)
    
    True
    return TrPrime, TfPrime, TffPrime, TbF #,  rPrime, fPrime, ffPrime, bF





def baseParsing(frames:list, page:object, totFrames:int, start:int=0) -> list:
    """This is the basic parsing of the images. Reads image and finds results and basic flagged results.

    :param frames: The array of frames: their image.
    :type frames: np.ndarray
    :param cropBounds: The bounds of the crop for the image.
    :type cropBounds: dict
    :param start: The starting index, only applicable from `firstImageError()`
    :type start: int
    :return: The results and the flags. See `R` and `F` in `gui.py` for more info.
    :rtype: (list, list[list])
    """
    res = np.empty([len(frames)],dtype=np.dtype('U15'))
    plainRes = []

    # Parsing the frames (from start)
    # for each frame, do OCR, then update 
    print("Parsing Frames...")
    _s1_total = (100/17)
    _s2_third = (100/17)/3
    _s3_expected = 2*(100/17)/3
    _s4_thirdFrames = ((100/17)/3)/totFrames
    _s5_2thirdFrames = (2*(100/17)/3)/totFrames
    # _s6 = ((100/17)/3)/len(frames)
    # _s7 = (2*(100/17)/3)/len(frames)
    _ss_current = 0
    _si_index = 0
    # frequency = totFrames//10
    # _s6_thread_expected = ((2*(100/17)/3))
    # step = frequency*(2*(100/17)/3)/totFrames
    # numRem = totFrames%frequency
    # print("\n\nFLAGS REM:", numRem, "\n\n")
    True


    for i, frame in enumerate(frames):
        if i < start: continue # by default nothing occurs, but if start has value, it skips the indexes below start
        index = i
        # * Effectively, start acts in a way to skip the frames below a certain index. 
        # * Therefore, (i+start) doesn't change *anything* except if start has value that's not 0.
        # * It makes things look complicated, but really is just i, usually.
        # if index > len(frames): continue
        # timeIt.Start(f"findText  -- Frame {i+1}")
        if mTest: print(1)
        if frame.any() == 0: # if the frame is empty, skip it
            res[index] = "None"
            continue
        res[index] = findText(frame) # finding the text in frame
        # timeIt.Stop()
        # timeIt.Start(f"flagWeird -- Frame {i+1}")
        # print(res[index])
        if mTest: print(5)
        plainRes.append(res[index])

        if mTest: print(6)
        if i == start: res[(index)] = flagWeird(res[(index)],"None") # there is no previous location if it's the first value, therefore potential for many issues 
        
        
        elif "FLAG" in res[(index)-1]: # If the previous result has a flag, pick the nearest result that doesn't have a flag.
            for e, item in enumerate(res): # * This is weird, and probably TODO: should be rewritten to use range and not have potential for as many errors. Use `finalPlausibilityCheck` in `OCR.py` as basic guidelines.
                if "FLAG" not in res[(index)-2-e]: 
                    res[(index)] = flagWeird(res[(index)],res[(index)-2-e])
                    break
        
        else: res[(index)] = flagWeird(res[(index)],res[(index)-1])  # otherwise, if no flag issues, check it to the previous result
        # timeIt.Stop()
        # timeIt.Start(f"flagging -- Frame {i+1}")
        # if "FLAG" in res[(index)]:  # if the result is flagged by flagWeird, add it to flags list
        #     flags.append(((index), res[(index)], frame))

        # * With the update of moving this to a new function, not sure if I need a loading bar.
        # if (index+1) % frequency == 0:
        #     # page.queue.put(step)
        #     # _s8 = page.barVal
        #     page.barVal += step
        #     _ss_current += step
        #     _si_index += 1
        #     print("flags - put", step, "- total val:", page.barVal)
        #     True
        # progress['value'] += len(frames)//100
        if mTest: print(7)
    # page.queue.put(((400*numRem)/15)/page.lenFrames)
    # page.barVal += numRem*(2*(100/17)/3)/totFrames
    # print("last flags - put", numRem*(2*(100/17)/3)/totFrames, "- total val:", page.barVal)
    if res[0] == "": res[0] = flag(res[0])
    return res

def batchFlags(res:list) -> list:
    # batching flags 
    # see `bF` in `gui.py` for more info

    # * Alternatives:
    # - List comprehension for indexes of flags in list
    #   - then look at which ones are next to eachother, and compare them
    # - See if re module works (or other module)


    """
    # TODO: For final, remove
    # started reintroducing batched flags -- currently not batching correctly
    bF = np.full((len(flags)),-1,dtype=np.int_)
    flagList = [(i,f"F{x[6:]}") for i, x in enumerate(flags) if "FLAG" in x]
    flen = len(flagList)
    
    timeIt.Start("BF Test Norm")
    for index,currentFlag in enumerate(flags):
        if "FLAG" in currentFlag:
            if index == 0: 
                bF[index] = index
                continue
            o = bF[:index]
            e = np.flipud(bF[:index])
            for prevIndex, prevFlag in enumerate(np.flipud(flags[:index])):
                batchedIndex =bF[index-prevIndex-1] 
                if bF[index-prevIndex-1] == -1 or prevIndex-index == 0:
                    # print("-1/Start"," - ","Curr-Prev Flags", currentFlag[6:], prevFlag[6:], " - ", "Prev Batched:", batchedIndex, " - ", "Current-Prev Indexes", index, prevIndex)
                    bF[index] = index
                    # print("Index", index, "is now", bF[index])
                    break
                elif prevFlag == currentFlag: 
                    oldInd = index-prevIndex-1
                    # print("Equals"," - ","Curr-Prev Flags", currentFlag[6:], prevFlag[6:], " - ", "Prev Batched:", batchedIndex, " - ", "Current-Prev Indexes", index, prevIndex)
                    bF[index] = bF[index-prevIndex-1]
                    # print("Index", index, "is now", bF[index])
                    break
                else: 
                    # print("Else"," - ","Curr-Prev Flags", currentFlag[6:], prevFlag[6:], " - ", "Prev Batched:", batchedIndex, " - ", "Current-Prev Indexes", index, prevIndex)
                    bF[index] = index
                    # print("Index", index, "is now", bF[index])
                    break

    timeIt.Stop()"""


    # * This is a more readable version of what batches is doing.
    """
    i = 0
    grouped = []

    for key, group in it.groupby(flags):
            number = next(group)
            elems = len(list(group)) + 1
            if number != "":
                grouped.append((key, elems, i))
            i += elems
    """

    l = len(res)
    i, grouped = 0, [] # i is the current index, grouped is the list of grouped flags
    for key, group in it.groupby(res): # for each group of flags
        number = next(group) # get the first number
        elems = len(list(group)) + 1 # get the number of elements in the group
        if "FLAG" in number: # if the number is a flag, add it to the grouped list
            grouped.append((elems, i)) # add the number of elements and the initial index to the grouped list
            # print("Added", number, " - ", str((elems, i)), "to grouped. -- key for ref", key)
        i += elems # increment the current index by the number of elements in the group
        # print(str(i) + "/" + str(l))
        True

    batches = np.asarray(grouped) # items are array of (repeats, initial index)



    # old, less understandble
    # batches = np.asarray(list((filter(lambda item: item[0] != "", ft.reduce(lambda lst,item: lst + [(item[0], item[1], lst[-1][1] + lst[-1][-1])], 
                        #  [(key, len(list(it))) for (key, it) in it.groupby(flags)], 
                        #  [(0,0,0)])[1:]))))
    
    

    True





    """
    TODO: For final, remove
    
    batchedFlags, repeats = {}, 0
    for i, flag in enumerate(flags): # for each flag in flag list
        if i == len(flags)-1:  # if the index is the last value,
            if repeats != 0: continue # if there currently are repeats, ignore it 
            else: # otherwise, add it to batched flags by itself
                batchedFlags[flag[1] + str(flag[0])] = [i]
                repeats = 0
                continue


        # Labelling key variables
        cF, nF, cFi, nFi = flag[1], flags[i+1]["val"], flag[0], flags[i+1]["rI"] 
        # These are key variables that are used through this comprehension.
        
        ### cF
        :var cF: The value of the current flag.
        :type cF: str

        ### nF
        :var nF: The value of the next flag.
        :type nF: str

        ### cFi
        :var cFi: The index of the current flag in `R`.
        :type cFi: int

        ### nFi
        :var nFi: The index of the next flag in `R`.
        :type nFi: int
        # 

        bcFi, bnFi = i, i+1
        # More key variables

        ### bcFi
        :var bcFi: The index of the current flag in `F`.
        :type bcFi: int

        ### bnFi
        :var bnFi: The index of the next flag in `F`.
        :type bnFi: int
        # 
        if nFi - cFi != 1: # if the next flag doesn't follow the current index in results, can't repeat, therefore they separate
            batchedFlags[cF + str(cFi)] = [bcFi]
            repeats = 0
            continue
        if i == 0 or nF != flags[i-1]["val"]: repeats = 0 # if the index of the current flag is 0 or the next flag isn't equivalent to the previous flag, repeats resets
                                                        # ? does this work??? TODO: check pls
        if nF == cF: # if the two values equal eachother, repeats is a go
            if repeats > 0: # if already repeating, find the correct key value and add the index of the *next* flag in `F` to batched Flags
                batchedFlags[cF + str(cFi-repeats)].append(bnFi)
            else: # if not repeating, add both indexes in `F` to a new key value based on the current value and index in `R`
                batchedFlags[cF + str(cFi)] = [bcFi, bnFi]
            repeats += 1
        else: # otherwise, they aren't the same, and the current flag gets its own batch, and repeats resets
            # ? might this have issues with like the last flag in each batch? idk TODO: check pls
            batchedFlags[cF + str(cFi)] = [bcFi]
            repeats = 0
            continue

    bF = np.ndarray((len(batchedFlags)),dtype=np.ndarray)
    for i, (key, val) in enumerate(batchedFlags.items()):
        bF[i] = np.array(val)

    return bF"""
    return batches

def finalChecks(res:list, frames:list, crop:tuple) -> tuple[list,list,list,list]:
    """This runs the final checks and balances to make sure the flags are correct and then sorts them into batches.

    :param res: The current list of results
    :type res: list
    :param flags: The current list of flags.
    :type flags: list[list]
    :param frames: The list of frames.
    :type frames: list
    :return: The updated list of flags, the updated list of results, and the dictionary of batched flags. See `F`, `R`, and `bF` in `gui.py` for more info.
    :rtype: (list[list], list, dict)
    """
    # Running final plausibility check to double check flags
    # timeIt.Start(f"Plaus Check")
    print("Plausibility Check...")
    res = finalPlausibilityCheck(res)
    # timeIt.Stop(
    
    flagList = [result if "FLAG" in result else "" for result in res]
    res = fixFlags(flagList, res)

    # ? this definitely feels redundant, but idk
    
    flags = np.array([result if "FLAG" in result else "" for result in res],dtype=np.dtype('U15'))
    shape = (len(frames),) + crop
    flaggedFrames = np.zeros((shape), dtype="uint8")
    for i in range(len(frames)):
        flaggedFrames[i,:] = frames[i] if "FLAG" in flags[i] else np.zeros((crop), dtype="uint8")
        # this is horrible, but i dont know what to do at this point
    # flaggedFrames = np.stack([flaggedFrames, curr_frame], axis=0) if flaggedFrames.size else curr_frame
    True

    # batchedFlags = batchFlags(flags)

    return res, flags, flaggedFrames #, batchedFlags

def parse(frames:np.ndarray, crop:tuple, page:object, totFrames:int) -> tuple[list, list[list], dict]:
    """This is the function that parses each frame for the two digit number for each frame, and then gives the results and flagged results (results that were read incorrectly)

    :param frames: The list of frames: their image.
    :type frames: np.array
    :param cropBounds: The bounds of the crop for the image.
    :type cropBounds: dict
    :return: The results, the flags, and the batched flags. See `R`, `F`, and `bF` in `gui.py` for more info.
    :rtype: (list, list[list], dict)
    """
    # Initialising variables, and beginning parsing
    res = baseParsing(frames=frames, page=page, totFrames=totFrames)

    # Final checks: Plausibilty check and sorting flags
    res, flags, fFrames = finalChecks(res, frames, crop)

    # This is protection from first flag errors (as well as just an overwhelming amount of errors.)
    # if len([flag for flag in flags if "FLAG" in flag]) == len(frames)-1:
    #     res, flags, fFrames = firstImgError(frames=frames, flags=flags, res=res,crop=crop)
    
    return res, flags, fFrames # batchedFlags # Returns the necessary values
    
def firstImgError(frames:list, flags: np.ndarray, res: np.ndarray,crop) -> tuple[list[list], list, dict]:
    """This occurs if over 50% of the frames are flagged. This probably means that there was a flag that occurred on the first result, causing EVERYTHING else to be flagged.
    Basically, the code, by default, assumes that first value is correct, so if it isn't everthing else gets flagged.
    This avoids that by trying to do the parsing again, but this time ignoring the first value, and re-adding it at the end.

    There is possibly a better solution to this problem, but I found this to be the most effective across all potentialities. 
        Theoretically if I held the original results, I could just say start checking flagWeird from 2nd result, but given that I don't do that and I don't recall if that would genuinely work, I desided on this.

    :param frames: The list of frames: their image.
    :type frames: list
    :param cropBounds: The bounds of the crop for the image.
    :type cropBounds: dict
    :param flags: The previous list of flags--only needed once
    :type flags: list
    :param res: The previous list of results--only needed once
    :type res: list
    :return: The new updated list of flags, updated list of results, and updated list of batched flags. See `R`, `F`, and `bF` in `gui.py` for more info.
    :rtype: (list[list], list, dict)
    """
    print("First image error:", res[0]) # I want it to say if a first image error occurs. At least in the console.

    firstFlag = 0 # This is the number of times the parsing has had to be rerun through. In the case of the first 3 getting flagged, this would end up on like 2 or 3, depending on when you check, but would allow for eventual correct parsing. Though it might take a while.
    flagged = [flag for flag in flags if "FLAG" in flag]
    True
    while len([flag for flag in flags if "FLAG" in flag]) > 0.5*len(frames): # while it's not working, redo and flag first image
        if firstFlag > 0: print("Another image error:", res[firstFlag])
        firstFlag +=1
        res = baseParsing(frames, start=firstFlag) # resetting variables and beginning reparsing
        
        # For the values that were skipped, re-add them to the correct index.
        for i in range(firstFlag):
            text = findText(frames[firstFlag-i-1])
            res[firstFlag-i-1] = f"FLAG: {text}"

        res, flags, fFrames = finalChecks(res, frames,crop) # Final checks: Plausibilty check and sorting flags
    return res, flags, fFrames # once less that 50% of frames are not flagged, return the data













