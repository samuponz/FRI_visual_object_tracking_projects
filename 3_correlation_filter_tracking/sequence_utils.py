import os
import glob

import numpy as np
import cv2


class VOTSequence():

    # constructor of the class where all the private and public attributes are initialized
    def __init__(self, dataset_path, sequence_name):
        self.sequence_path = os.path.join(dataset_path, sequence_name)  # path where the frames are stored
        self.frames = []  # list of frames
        self.gt = []  # list of groundtruth positions
        self.window_name = ''  # Name of the Figure where the video is reproduced
        self.load_sequence()  # execute this method when the object is initialized

    # method that finds the frames in the folder and stores them in a list, it also stores the groundtruth positions
    # this is ran at every instantiation of the VOT sequence object
    def load_sequence(self):
        frames_path = os.path.join(self.sequence_path, 'color')  # it creates the string 'main folder' + 'color'
        if not os.path.exists(frames_path):  # if the string isn't a folder in the main folder
            frames_path = self.sequence_path   # it sets the folder of the frames as the main folder
            # (for VOT folder is always like this)

        self.frames = sorted(glob.glob(os.path.join(frames_path, '*.jpg')))  # saves the images in the main folder in
        # a list called 'frames'. Since the frames are named with numbers it uses glob to sort them)
        self.gt = self.read_groundtruth(os.path.join(self.sequence_path, 'groundtruth.txt'))  # saves the coordinates
        # of the groundtruth bounding shapes in a list, using the method 'read_groundtruth'

    # method that simply returns a frame given the index, which is an external parameter
    def frame(self, frame_idx):
        return self.frames[frame_idx]

    # method that simply returns the length of the list 'frames'
    def length(self):
        return len(self.frames)

    # method that given a path of a .txt file returns a list with the groundtruth regions coordinates in every frame
    def read_groundtruth(self, file_path):
        with open(file_path, 'r') as gt_file:  # reads the .txt file
            gt_ = gt_file.readlines()
            return [[float(el) for el in line.strip().split(',')] for line in gt_]  # returns a list where every element
            # is a list of groundtruth coordinates in a frame.
            # strip() method returns a copy of the string by removing both the leading and the trailing characters
            # (based on the string argument passed)

    # Annotation (bounding shape) methods -----------------------------------------------------------------------------

    # method that given a frame index returns the annotation/region (list of coordinates of the bounding shape) of the
    # groundtruth position.
    def get_annotation(self, frame_idx, type='rectangle'):
        if type == 'rectangle':
            return self.convert_region(self.gt[frame_idx], type)
        elif type == 'polygon':
            return self.convert_region(self.gt[frame_idx], type)
        else:
            print('Error: Unknown annotation format.')
            exit(-1)

    # method that given a region and the type of that region, converts the region form one type to the other.
    def convert_region(self, region, type):
        # if the number of points in the region list already matches the type of region, simply return the region
        if (len(region) == 4 and type == 'rectangle') or (len(region) == 8 and type == 'polygon'):
            return region
        # from polygon to rectangle
        elif len(region) == 8 and type == 'rectangle':
            # convert from polygon to rectangle using min-max rectangle
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            return [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]
        # from rectangle to polygon
        elif len(region) == 4 and type == 'polygon':
            x0 = region[0]
            y0 = region[1]
            x1 = x0 + region[2] - 1
            y1 = y0 + region[3] - 1
            return [x0, y0, x1, y0, x1, y1, x0, y1]
        else:
            print('Error: Cannot convert region.')
            exit(-1)

    # This method computes the IoU between the gt bbox and the estimated bbox. If the regions are polygons, they are
    # converted to bounding boxes.
    def overlap(self, region1, region2):
        # simplified overlap: region1 and region2 are converted into axis-aligned bounding boxes
        bb1 = self.convert_region(region1, type='rectangle')
        bb2 = self.convert_region(region2, type='rectangle')
        # coordinates of the intersect
        xA = max(bb1[0], bb2[0])
        yA = max(bb1[1], bb2[1])
        xB = min(bb1[0] + bb1[2] - 1, bb2[0] + bb2[2] - 1)
        yB = min(bb1[1] + bb1[3] - 1, bb2[1] + bb2[3] - 1)
        # area of the intersect
        intersect_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # areas of both bounding boxes
        area1 = bb1[2] * bb1[3]
        area2 = bb2[2] * bb2[3]
        # IoU = intersect over union - classic metric for accuracy in object recognition tasks
        return intersect_area / float(area1 + area2 - intersect_area)

    # drawing functions -----------------------------------------------------------------------------------------------

    # method that creates the window where to play the video
    def initialize_window(self, window_name):
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

    # method that draws the annotation shape given: frame, type of shape/region, color and line width for the bbox
    def draw_region(self, img, region, color, line_width):
        if len(region) == 4:  # if the region is a list of 4 elements
            # rectangle
            tl = (int(round(region[0])), int(round(region[1])))  # top left corner
            br = (int(round(region[0] + region[2] - 1)), int(round(region[1] + region[3])))  # bottom right corner
            cv2.rectangle(img, tl, br, color, line_width)  # draw a rectangle on the given image, since nothing is
            # returned i think it directly plots the img with the rectangle
        elif len(region) == 8:  # if the region is a list of 8 elements
            # polygon
            pts = np.round(np.array(region).reshape((-1, 1, 2))).astype(np.int32)
            cv2.polylines(img, [pts], True, color, thickness=line_width, lineType=cv2.LINE_AA)
        else:
            print('Error: Unknown region format.')
            exit(-1)

    # method that draws white text on the image in a black box
    def draw_text(self, img, text, text_pos):
        font = cv2.FONT_HERSHEY_PLAIN
        text_sz = cv2.getTextSize(text, font, 1, 1)
        tl_ = (text_pos[0] - 5, text_pos[1] + 5)
        br_ = (text_pos[0] - 5 + text_sz[0][0] + 10, text_pos[1] - 5 - text_sz[0][1])
        cv2.rectangle(img, tl_, br_, (0, 0, 0), cv2.FILLED)
        cv2.putText(img, text, text_pos, font, 1, (255, 255, 255), 1, cv2.LINE_AA, False)

    def show_image(self, img, delay):
        cv2.imshow(self.window_name, img)
        cv2.waitKey(delay)
