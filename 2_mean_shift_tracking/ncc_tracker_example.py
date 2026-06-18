import numpy as np
import cv2

from ex2_utils import Tracker


# NORMALIZED CROSS CORRELATION TRACKER --------------------------------------------------------------------------------
# NCCTracker inherits from the Tracker class
class NCCTracker(Tracker):

    def initialize(self, image, region):

        if len(region) == 8:  # if the region is a polygon, it converts it to a bounding box
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            # tl coordinates of the square bounding box plus width and height of the box
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]

        self.window = max(region[2], region[3]) * self.parameters.enlarge_factor # creates a 'window' attribute taking
        # the max(width, height) and multiplying it by the enlarge factor. This is a window with a enlargedff size in
        # respect to the template, it is the region where we search for the new position of the target (computing the
        # cross-correlation).

        # takes into account if the region exits the bounds of the image
        left = max(region[0], 0)
        top = max(region[1], 0)
        right = min(region[0] + region[2], image.shape[1] - 1)
        bottom = min(region[1] + region[3], image.shape[0] - 1)

        # VISUAL MODEL = TEMPLATE, cropped from the image
        # creates a template by cropping the image according to the values: top, bottom, left, right
        self.template = image[int(top):int(bottom), int(left):int(right)]
        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)  # Estimated position of the target
        # In the first frame it is obviously the position of the target template.
        self.size = (region[2], region[3])  # size of the template

    def track(self, image):
        # In this method, the template is obviously not updated, the target and remains the same. What is updated is the
        # estimated position of the target in this frame, which is computed with the cross-correlation between the image
        # and the template.
        # The estimated position is returned

        # Coordinates ot the window centered at the estimated position, where we search fot the new position of the
        # template
        left = max(round(self.position[0] - float(self.window) / 2), 0)
        top = max(round(self.position[1] - float(self.window) / 2), 0)
        right = min(round(self.position[0] + float(self.window) / 2), image.shape[1] - 1)
        bottom = min(round(self.position[1] + float(self.window) / 2), image.shape[0] - 1)


        if right - left < self.template.shape[1] or bottom - top < self.template.shape[0]:
            return [self.position[0] + self.size[0] / 2, self.position[1] + self.size[1] / 2, self.size[0], self.size[1]]

        cut = image[int(top):int(bottom), int(left):int(right)]  # crops the image in the window

        matches = cv2.matchTemplate(cut, self.template, cv2.TM_CCOEFF_NORMED)  # computes the correlation between window
        # and the template

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matches)  # gets the max value location (local coordinates
        # in the window)

        # Update the position of the target in this frame, it will be used in the next call
        self.position = (left + max_loc[0] + float(self.size[0]) / 2, top + max_loc[1] + float(self.size[1]) / 2)

        # returns the coordinates of the new estimated region
        return [left + max_loc[0], top + max_loc[1], self.size[0], self.size[1]]


# class of the parameters of the tracker, here we generate all the parameters with a custom or a default value. In this
# case, there is only one default parameters, which is 'enlarge_factor".
# Remember that different constructors can be created just by changing the number of parameters for every constructor!
class NCCParams():
    def __init__(self):
        # we don't have external parameters, we just create a default parameter 'enlarge_factor' and we set it to 2
        self.enlarge_factor = 2

