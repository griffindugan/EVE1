"""

Author: Griffin Dugan
Brandeis Memory & Cognition (Wingfield) Lab
EVE1 OCR Program

Description: Required file of OCR functions for gui.py to run. This file is used to extract text from images, and then to optimise that text to be used in the main program.
"""


import pytesseract
import numpy as np # TODO, never used???
import cv2

# purely for timeIt()
import time 
import tracemalloc

mTest = False

# Dictionaries for fixing consistent errors
# TODO: For Final, fix up dictionaries to only include necessary
nonNumChar = {
    "?" : "0",
    "?" : "1",
    "?" : "2",
    "?" : "3",
    "a" : "4",
    "?" : "5",
    "?" : "6",
    "?" : "7",
    "?" : "8",
    "Q" : "9"
}

# TODO: For Final, either add functionality for this or scrap it 
tooManyChar = {
    "?" : "0",
    "?" : "1",
    "?" : "2",
    "?" : "3",
    "a" : "4",
    "?" : "5",
    "?" : "6",
    "?" : "7",
    "?" : "8",
    "Q" : "9"
}

class check():
    def __init__(self, memory:bool=False) -> None:
        self.start, self.function, self.memory = float, str, memory
        self.difference = float
        self.memCurr, self.memPeak = int, int

    def Start(self, funct:str):
        self.function, self.start = funct, time.time()
        # if memory != None: self.memory = memory
        if self.memory: tracemalloc.start()

    def Stop(self,toPrint:bool=True):
        self.memCurr, self.memPeak = tracemalloc.get_traced_memory()[0], tracemalloc.get_traced_memory()[1]
        self.function, self.difference, self.memCurr, self.memPeak = self.organise(self.function, self.difference, self.memCurr, self.memPeak)
        if toPrint: print(f"--- {self.function} --- {self.difference} seconds --- {"" if not self.memory else f"Memory Current --- {self.memCurr} --- Memory Peak --- {self.memPeak}"} ---")
        tracemalloc.stop()

    def organise(self,function,difference,memCurr,memPeak):
        function = str(function) + (" "*(25-len(str(function))))
        difference = ("%.17f" % round(float((time.time() - self.start)),10)).rstrip('0').rstrip('.')
        if len(difference) > 12: difference = round(float(difference),10)
        difference = str(difference) + (" "*(12-len(str(difference))))
        if self.memory: 
            memCurr = str(memCurr) + (" "*(7-len(str(memCurr))))
            memPeak = str(memPeak) + (" "*(7-len(str(memPeak))))
            return function, difference, memCurr, memPeak
        else:
            return function, difference, 1, 1
            
            

    def convScient(number:float):
        pass

    def runAverage(self, runs:int, functName:str, function, *args, printEach:bool=True, printResults:bool=True):
        difference, cMem, pMem = 0.0, 0, 0
        for i in range(int(runs)):
            self.Start(funct=f"{functName} {i+1}")
            function(*args)
            self.Stop(toPrint=printEach)
            difference += float(self.difference)
            cMem += int(self.memCurr)
            pMem += int(self.memPeak)
        self.difference, self.memCurr, self.memPeak = difference/runs, round(cMem/runs), round(pMem/runs)
        # print(f"--- {functName} --- {self.difference} seconds --- Memory Current --- {self.memCurr} --- Memory Peak --- {self.memPeak}")
        functionName, self.difference, self.memCurr, self.memPeak = self.organise(functName,self.difference,self.memCurr,self.memPeak)
        if printResults: print(f"-------------------------------------------------------------------------------------------------------------")
        if printResults: print(f"--- {functionName} --- {self.difference} seconds --- Memory Current --- {self.memCurr} --- Memory Peak --- {self.memPeak} ---")
        else: return f"--- {functionName} --- {self.difference} seconds --- Memory Current --- {self.memCurr} --- Memory Peak --- {self.memPeak} ---"

    def Compare(self, runs:int, functNames:tuple, functions:tuple, *args):
        # not working
        for i, function in enumerate(functions):
            self.runAverage(runs=runs, functName=functNames[i], function=function, printEach=False,printResults=True, *args[i])
            

        

timeIt = check(memory=True)


def findText(image: list[list]) -> str:
    """Finds text in image. Intended to be used to find two digit large number in centre of screen. It crops out all but the designated location, so moving camera will not work.

    :param image: The cv2 image
    :type image: list[list]
    :param cropBounds: The bounds for cropping the image. ({"H":[y0,y1],"W":[x0,x1]})
    :type cropBounds: dict
    :return: The text in the image
    :rtype: str
    """
    # Cropping Image
    # timeIt.startT("Crop")
    
    if mTest: print(2)
    # timeIt.stop()

    # timeIt.startT("Image Manip")

    # Making the image black & white and blurred
    # Using: Greyscale, Gaussian blur, Otsu's threshold
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # greyscale
    blur = cv2.GaussianBlur(grey, (5,5), 0) # blurred
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] # this makes the contours pop out

    # Inverting Image
    invert = 255 - thresh

    # timeIt.stop()

    # timeIt.startT("OCR")

    # Extracting Text
    if mTest: print(3)
    data = pytesseract.image_to_string(invert, lang='eng', config='--psm 6 -c tessedit_char_whitelist=0123456789')
    # print(data)

    # timeIt.stop()

    # * If you want to show the image, uncomment the next 2 lines.
    # cv2.imshow('invert', invert)
    # cv2.waitKey()
    if mTest: print(4)
    return data.replace("\n","")  # returning text, without a new line

def contains_nonNum(text: str) -> bool:
    """Returns True if there's a character that's not a number in the string. False if all numbers.

    :param text: Inputed string
    :type text: str
    :return: True if character that's not a number, False if all numbers
    :rtype: bool
    """
    return not all(char in "1234567890" for char in text)

def plausibleCheck(text: str, prev_text: str) -> bool:
    """Returns whether the number is in the plausible range (of 3) based on the previous number.

    :param text: The current number
    :type text: str
    :param prev_text: The previous number
    :type prev_text: str
    :return: Whether or not the number is plausible (True: Plausible, False: Not Plausible).
    :rtype: bool
    """ 
    # TODO might be worthwhile to put a try except catch here to just flag stuff that causes errors
    try:
        if prev_text == "": return False
        else: return not (abs(int(text)-int(prev_text)) > 2)
    except ValueError:
        return False

def flag(text: str) -> str:
    """Flags for user input.

    this feels like it shouldn't be a function, but oh well

    :param text: The flagged text
    :type text: str
    :return: The flagged text, *but with `FLAG: ` in front of it...
    :rtype: str
    """
    return f"FLAG: {text}" 

def nonNumbers(text: str, prev: str) -> str:
    """This optimises an intended two digits of numbers to be truely numerical. 
    It uses a predefined dictionary to substitute out the non-numerical digits with what they've been known to mess up.

    :param text: The flagged result
    :type text: str
    :param prev: The previous result
    :type prev: str
    :return: The attempted fixed result, but still going through double checking for more weird.
    :rtype: str
    """
    optimised = ""
    for i,character in enumerate(text):
        if contains_nonNum(character): # TODO: list comprehension this
            for number in nonNumChar: 
                if text[i] == number: # if the text starting at the index to the length of the read number is the same as the read number, it's the number that the read number correlates to.
                    optimised += nonNumChar[number]
        else: optimised += character
    return flagWeird(optimised, prev)

def tooManyLetters(text: str, prev: str) -> str:
    """If there are too many letters in the flagged result, try to fix that.

    :param text: The flagged result
    :type text: str
    :param prev: The previous result
    :type prev: str
    :return: The attempted to be fixed result
    :rtype: str
    """
    # if prev == "None" and contains_nonNum(text): return nonNumbers(text[0:2], prev) # dealing with first result being potentially flagged
    if contains_nonNum(text):
        if len("".join([number for number in text if number in "1234567890"])) != 2: return flag(text) # if when removing non-numerical, it ends up being all non-numerical, just flag it.
        if prev == "None": return "".join([number for number in text if number in "1234567890"]) # this is reckless, given that it could be a wrong first value, but that's what first flag error function is for
        elif plausibleCheck("".join([number for number in text if number in "1234567890"]),prev_text=prev): # if there are some numbers left, see if that is plausible, and if so that's prob right. If not, flag it.
            return "".join([number for number in text if number in "1234567890"])
        else: return nonNumbers(text, prev)
    else:
        return flag(text)

def flagWeird(text: str, prev_text: str) -> str:
    """Function for flagging weird results.

    :param text: The result
    :type text: str
    :param prev_text: The previous result
    :type prev_text: str
    :return: The result, either flagged or normal.
    :rtype: str
    """

    if len(text) != 2: \
        return (tooManyLetters(text, prev_text) if len(text) > 2 else flag(text))
    elif contains_nonNum(text):
        return nonNumbers(text, prev_text)
    # elif prev_text != "None" and not plausibleCheck(text, prev_text):
    #     return flag(text)
    else:
        return text

def finalPlausibilityCheck(res: np.ndarray) -> np.ndarray:
    """The final plausibility check to check before and after of each result in the list of results
    This could stand to have some work done on it.

    :param res: The array of results
    :type res: np.ndarray
    :return: The updated array of results
    :rtype: np.ndarray
    """
    for i, result in enumerate(res): # go through each result
        before, after = "", "" # Before and after results
        if i == 0 or i == len(res)-1 or "FLAG" in result: continue # if first or last, skip, OR if there's already a flag in the result
        if "FLAG" in res[i-1]: # if there's a flag in the previous result, find the nearest result that isn't flagged.
            for e, item in enumerate(res): # TODO: switch to range for more safe
                if "FLAG" not in res[i-2-e]: 
                    before = res[i-2-e]
                    break
        else: before = res[i-1]
        if "FLAG" in res[i+1]: # if there's a flag in the next result, find the nearest result that isn't flagged. 
            if i == len(res)-2: 
                res[i] = flag(result)
                continue
            # TODO: fix this range function to be capped at the end of res, should solve errors here
            q = res[i+1:]
            for e in range(len(res[i+1:])): # ! there's an error that occurs around here every once and a while. Not sure what the cause is still.
                if "FLAG" in res[i]: break
                if i == len(res)-2-e: 
                    res[i] = flag(result)
                    after = res[i+1]
                elif "FLAG" not in res[i+2+e]: 
                    after = res[i+2+e]
                    break
        else: after = res[i+1]

        if not any("FLAG" in item for item in (res[i], before, after)):
            # If the result isn't plausible with the surrounding results, flag it.
            if not plausibleCheck(result, before) and not plausibleCheck(after,result):
                res[i] = flag(result)
            elif not plausibleCheck(result, before) or not plausibleCheck(after,result):
                res[i] = flag(result)
    return res # return new list of results

def fixFlags(flags: list, res: np.ndarray) -> tuple[np.ndarray]:
    # if past 5 or so (at least I think it's 5) flags are the same thing, make them unflagged.
    for i, flag in enumerate(flags):
        if flag == "": continue # therefore, not flagged
        if contains_nonNum(flag[6:]) or len(flag[6:])!=2: continue # If the flagged result contains non numbers in it or if it's not 2 digits, then therefore it shouldn't be accepted
        # this is a failsafe for many 4s in a row
        # !!!! ISSUE HERE -- maybe?
        if i > 0:
            if flag[6:] == res[i-1]: # if it's not the first result, and it's the same as previous, unflagged result, unflag it.
                # Assists issue where the value before a weird result gets flagged. *assists* only because no effective fix
                res[i] = flag[6:]
        if i < 4: continue # if there aren't 5 indexes before it, wait for there to be
        repeats = 0
        for e in range(5): # ? There's something here that fits strangely with the previous correction, like I think it would prevent repeats. TODO: fix?
            if flag == flags[i-e-1]: repeats += 1 # if the flag equals the same as the previous index - e (0-4), add a repeat counter
            else: break # if it doesn't at any point, break out of for loop
            if repeats == 5: # if repeats hits 5, therefore the last 5 are all the same, so unflag all of them
                for r in range(5): 
                    res[i-r] = flag[6:]
    return res

def sortFlags(flags: list, res: list) -> tuple[list, list, dict]:
    """Sorts through flags and fixes incorrect flags, and then batches the flags together.

    :param flags: List of flags
    :type flags: list
    :param res: List of results
    :type res: list
    :return: The updated list of flags, the updated list of results, and the dictionary of batched flags.
    :rtype: list, list, dict
    """
    # timeIt.Start(f"Sort Flags")
    # if past 5 or so (at least I think it's 5) flags are the same thing, make them unflagged.
    for i, flag in enumerate(flags):
        if contains_nonNum(flag[1][6:]) or len(flag[1][6:])!=2: continue # If the flagged result contains non numbers in it or if it's not 2 digits, then therefore it shouldn't be accepted
        # this is a failsafe for many 4s in a row
        # !!!! ISSUE HERE -- maybe?
        if flag[0] > 0:
            if flag[1][6:] == res[flag[0]-1]: # if it's not the first result, and it's the same as previous, unflagged result, unflag it.
                # Assists issue where the value before a weird result gets flagged. *assists* only because no effective fix
                res[flag[0]] = flag[1][6:]
        if i < 4: continue # if there aren't 5 indexes before it, wait for there to be
        repeats = 0
        for e in range(5): # ? There's something here that fits strangely with the previous correction, like I think it would prevent repeats. TODO: fix?
            if flag[1] == flags[i-e-1][1]: repeats += 1 # if the flag equals the same as the previous index - e (0-4), add a repeat counter
            else: break # if it doesn't at any point, break out of for loop
            if repeats == 5: # if repeats hits 5, therefore the last 5 are all the same, so unflag all of them
                for e in range(5): 
                    res[flags[i-e][0]] = flag[1][6:]


    # updating flags based on the changes
    # This can be done through list comprehension but is a hell of a lot more messy, so I just left it as this.
    oldFlags = list(zip(*flags)) # unzips the flags into individual lists of indexes in `R`, flag values, and images
    indexesR, indexesF = [], []
    # for each index in `oldFlags[0]` (list of indexes in `R`), if the index (`x`) isn't in `indexesR` list, add it and add the index in `oldFlags[0]` (`i`) to `indexesF`.
    # this is weird, but basicically, below I prune out any duplicate indexes, so this declares which indexes are unique, based on `R`, and adds their index in `F` to a list.
    for i,x in enumerate(oldFlags[0]):
        if x not in indexesR:
            indexesF.append(i)
            indexesR.append(x)

    # Redefining flags
    notableIndexes, newFlags = [], []
    for i,flag in enumerate(oldFlags[0]):
        if "FLAG" in res[flag]: notableIndexes.append(i) # if the flagged index is currently flagged in results, add that index
    for index in notableIndexes: # then for each index in that now flagged set of indexes, go through and if the index isn't already in the previous indexes, add it
        if index not in indexesF: continue # ? this feels weird to me, not sure if this is right. TODO: Maybe check back in on this at some point?
        newFlags.append(flags[index])

    # batching flags 
    # see `bF` in `gui.py` for more info
    batchedFlags, repeats = {}, 0
    for i, flag in enumerate(newFlags): # for each flag in flag list
        if i == len(newFlags)-1:  # if the index is the last value,
            if repeats != 0: continue # if there currently are repeats, ignore it 
            else: # otherwise, add it to batched flags by itself
                batchedFlags[flag[1] + str(flag[0])] = [i]
                repeats = 0
                continue


        # Labelling key variables
        cF, nF, cFi, nFi = flag[1], newFlags[i+1][1], flag[0], newFlags[i+1][0] 
        """These are key variables that are used through this comprehension.
        
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
        """

        bcFi, bnFi = i, i+1
        """More key variables

        ### bcFi
        :var bcFi: The index of the current flag in `F`.
        :type bcFi: int

        ### bnFi
        :var bnFi: The index of the next flag in `F`.
        :type bnFi: int
        """
        if nFi - cFi != 1: # if the next flag doesn't follow the current index in results, can't repeat, therefore they separate
            batchedFlags[cF + str(cFi)] = [bcFi]
            repeats = 0
            continue
        if i == 0 or nF != newFlags[i-1][1]: repeats = 0 # if the index of the current flag is 0 or the next flag isn't equivalent to the previous flag, repeats resets
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
    # timeIt.Stop()
    return newFlags, res, batchedFlags
    


def optimiseText(text: str) -> str:
    """UNNECESSARY
    TODO: For Final, remove
    
    
    Clears out unintended text from image, and returns the intended two digit number.

    :param text: The text found using OCR.
    :type text: str
    :return: The correct two digit number.
    :rtype: str
    """
    
    



    # Fixing the numbers, and making sure they are correct.
    optimised, skipnext = "", 0 # The new numbers and the number of times to skip digits.

    # # for each of the numbers, check to see if the next set of digits in text is expected
    # for i,character in enumerate(text):
    #     if skipnext > 0: continue # If there is a number read as 2 digits, skip the second digit.
    #     for number in characters: 
    #         if text[i:i+(len(number))] == number: # if the text starting at the index to the length of the read number is the same as the read number, it's the number that the read number correlates to.
    #             optimised += characters[number]
    #             skipnext = len(number)-1

    return optimised


def test():
    filename = 'testImage.jpg'
    image = cv2.imread(filename)
    text = findText(image=image, cropBounds={"H":[235,300],"W":[890,990]})
    # print(text)

# test()
# timeIt.runAverage(runs=50,functName="test",function=test)

def TfindText(image: list[list]) -> str:
    """Finds text in image. Intended to be used to find two digit large number in centre of screen. It crops out all but the designated location, so moving camera will not work.

    :param image: The cv2 image
    :type image: list[list]
    :param cropBounds: The bounds for cropping the image. ({"H":[y0,y1],"W":[x0,x1]})
    :type cropBounds: dict
    :return: The text in the image
    :rtype: str
    """
    # Cropping Image
    # timeIt.startT("Crop")
    
    # y1, y2, x1, x2 = int(cropBounds["H"][0]), int(cropBounds["H"][1]), int(cropBounds["W"][0]), int(cropBounds["W"][1]) # Defining crop based on cropBounds param
    # cropped_image = image[y1:y2,x1:x2]
    if mTest: print(2)
    # timeIt.stop()

    # timeIt.startT("Image Manip")

    # Making the image black & white and blurred
    # Using: Greyscale, Gaussian blur, Otsu's threshold
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # greyscale
    blur = cv2.GaussianBlur(grey, (5,5), 0) # blurred
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] # this makes the contours pop out

    # Inverting Image
    invert = 255 - thresh

    # timeIt.stop()

    # timeIt.startT("OCR")

    # Extracting Text
    if mTest: print(3)
    data = pytesseract.image_to_string(invert, lang='eng', config='--psm 6')
    # print(data)

    # timeIt.stop()

    # * If you want to show the image, uncomment the next 2 lines.
    # cv2.imshow('invert', invert)
    # cv2.waitKey()
    if mTest: print(4)
    return data.replace("\n","")  # returning text, without a new line