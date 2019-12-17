#imports
from enum import Enum

"""
For passing the information of each character template.
-> (location of top-left, width, height, label)

label is the string mark of Number and Letter (0 ~ 9, A ~ Z or a ~ z), case sensitive.
"""


# return result information
class ResultInfo(object):
    def __init__(self, step, success, timeStamp, projectName, imageName, resultImagePath):
        self.step = step
        self.success = success
        self.timeStamp = timeStamp
        self.projectName = projectName
        self.imageName = imageName
        self.resultImagePath = resultImagePath


class TemplateBox(object):
    # (x, y) is top-left coordinate
    def __init__(self, x, y, width, height, content, rotate):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.content = content
        self.rotate = rotate

    def get_label(self):
        return self.content

    def get_location(self):
        return (self.x, self.y)

    def get_size(self):
        return (self.width, self.height)

# User option state machine
class ParamAlignment(object):
    def __init__(self, max_features, good_match_percent):
        self.max_features = max_features
        self.good_match_percent = good_match_percent

class ParamPreProcessing():
    def __init__(self, kernel_blur, threshold_binary):
        self.kernel_blur = kernel_blur
        self.threshold_binary = threshold_binary

class recognition_result(object):
    def __init__(self, character, defect, blur):
        self.character = character
        self.defect = defect
        self.blur = blur

class object_recogniton_result(recognition_result, TemplateBox):
    def __init__(self, character, defect, blur, x, y, width, height, content, rotate, count, form):
        recognition_result.__init__(self, character, defect, blur)  # returned recognition result
        TemplateBox.__init__(self, x, y, width, height, content, rotate)  # original template box
        self.count = count  # inputTemplate count
        self.form = form  # form = digit, upper, lower
