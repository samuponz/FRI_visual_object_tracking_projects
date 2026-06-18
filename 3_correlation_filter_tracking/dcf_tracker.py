import math

import numpy as np
import cv2
from my_utils import Tracker, get_patch, create_cosine_window, create_gauss_peak
from matplotlib import pyplot as plt

# Parameters that need to be shared between the two methods are stored as class attributes
# ATTRIBUTES:
# self.target_position (computed at the end of the tracking and used at the beginning of the next tracking)
# self.search_region_size (used to set the size of multiple quantities like
#                   - the gaussian ideal response,
#                   - the patch (cropped image),
#                   - the cosine window that multiplies the patch for reducing the effect of the cyclic shift
# The following 4 are used to create the matrices of shifts that ware used to compute the correct target position
# self.w2
# self.h2
# self.x
# self.y

# For filter application and update
# self.h (the correlation response between the the search region and the ideal gaussian response)
# self.H_conj (the complex conj of the filter estimation at the current frame)
# self.G (the ideal response, it is computed one time and stored)
# self.c (the cosine window used to modulate the search region

# For debugging:
# self.h
# self.r
# self.fc

from sequence_utils import VOTSequence


def freq_to_spatial(H_conj):
    # Given a 2D spectrum it recovers the corresponding image in the spatial domain
    H = np.conj(H_conj)
    h = np.fft.ifft2(H)
    # x can be complex, with a really small imaginary part. We ignore the imaginary part
    h = np.real(h)
    return h


# DISCRIMINATIVE CORRELATION FILTER TRACKER ---------------------------------------------------------------------------
# DCF TRACKER inherits from the Tracker class
class DCFTracker(Tracker):

    def filter_learning(self, image_gray):
        # Computation of the filter given the image and the desired response

        # Get patch 'f', from the region coordinates
        f, mask = get_patch(image_gray, self.target_position, self.region_size)
        # If the region goes out of the image, don't update the filter
        if 0 in mask:
            return self.H_conj
        # multiply patch with the cosine window c
        fc = np.multiply(f, self.c)
        # compute F, fft of (f*c)
        F = np.fft.fft2(fc)
        # compute complex conjugate of F
        F_conj = np.conj(F)
        # compute H* (it will be complex conjugate)
        eps = 1e-10  # make sure the denominator is not zero
        H_conj = np.divide(np.multiply(self.G, F_conj), np.multiply(F, F_conj) + eps + self.parameters.lmbd)
        return H_conj

    def filter_application(self, image_gray):  # localization
        # Get patch 'f', from the region coordinates
        f, _ = get_patch(image_gray, self.target_position, self.region_size)
        # multiply patch with the cosine window c
        self.fc = np.multiply(f, self.c)
        # compute F, fft of (f*c)
        F = np.fft.fft2(self.fc)
        # compute the spectrum of the correlation response (power spectral density)
        R = np.multiply(self.H_conj, F)
        # compute the correlation response (spatial domain obviously
        r = np.fft.ifft2(R)
        # trow away imaginary part
        r = np.real(r)
        # Center the correlation response in the center of the image
        self.r = np.roll(r, (self.h2, self.w2), (0, 1))
        # Find index in the array corresponding to the max of the correlation response
        index = np.argmax(self.r)  # with one argument only, argmax return the position in the flatten 1D array
        # Create empty np array for the value of the shift
        shift = np.empty(len(self.target_position), dtype=float)
        shift[0] = self.x.flatten(order='C')[index]
        shift[1] = self.y.flatten(order='C')[index]
        return shift

    def initialize(self, image, region):
        # image is the initialization frame
        # region is a list of 4 coordinates identifying the target region

        # convert image to grayscale
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Get region size (make sure the region is odd sized)
        self.region_size = (int((np.ceil((region[2]*self.parameters.enlarge_factor) // 2) * 2 + 1)),
                            int((np.ceil((region[3]*self.parameters.enlarge_factor) // 2) * 2 + 1)))
        # NOTE: THE REGION COORDINATES ARE STORED WITH CARTESIAN INDEXING (x, y), a region is [x, y, w, h]
        # We are working with images(matrices) that uses MATRIX INDEXING (i-row, j-col)
        # therefore the width of the region goes over the cols, the height goes over the rows

        # Get and store half dimensions of the region
        # (to compute the roll of the correlation response in every learning execution)
        self.w2 = int(np.floor(self.region_size[0] / 2))
        self.h2 = int(np.floor(self.region_size[1] / 2))
        # Get and store matrices of shifts centered in the center (to compute the shift in the localization step)
        self.x, self.y = np.meshgrid(np.arange(-self.w2, self.w2 + 1), np.arange(-self.h2, self.h2 + 1))

        # Find the central position of the given ground truth target region (center of the target)
        self.target_position = np.array([round(region[0] + region[2] / 2), round(region[1] + region[3] / 2)])

        # Compute the ideal response g
        g = create_gauss_peak(self.region_size, self.parameters.sigma)
        # Compute G = fft(g)
        self.G = np.fft.fft2(g)
        # Compute c (cosine window)
        self.c = create_cosine_window(self.region_size)

        # Filter leaning given the target in the first frame
        self.H_conj = self.filter_learning(image_gray)

        # Compute the filter in the spatial domain to visually observe the result
        self.h = freq_to_spatial(self.H_conj)

    def track(self, image):
        # convert image to grayscale
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Localization: given self.H* the position of the target IN THE NEIGHBORHOOD is computed
        # Given the location of the max in the neighborhood it is possible to compute the shift from the last position
        shift = self.filter_application(image_gray)

        self.target_position[0] += shift[0]
        self.target_position[1] += shift[1]

        # Update: given the new region centered in the new estimated position, compute the filter H
        H_conj = self.filter_learning(image_gray)
        # Temporal averaging using alpha (update speed)
        self.H_conj = ((1-self.parameters.alpha)*self.H_conj) + (self.parameters.alpha*H_conj)
        # Compute the filter in the spatial domain to observe the result
        self.h = freq_to_spatial(self.H_conj)

        # fig, ax = plt.subplots(1, 3)
        # ax[0].imshow(self.fc, cmap='gray')
        # ax[0].set_title("Search\nregion")
        # ax[1].imshow(self.r, cmap='gray')
        # ax[1].set_title("Correalation\nresponse")
        # ax[2].imshow(self.h, cmap='gray')
        # ax[2].set_title('Filter\nresponse')
        # plt.show()

        # print('Shift:')
        # print(shift[0])
        # print(shift[1])

        # Returns the coordinates of the new estimated region
        return [self.target_position[0] - self.region_size[0] / 2.0,
                self.target_position[1] - self.region_size[1] / 2.0,
                self.region_size[0],
                self.region_size[1]]


# class of the parameters of the tracker, here we generate all the parameters with a custom or a default value.
class DCFParams():
    def __init__(self, enlarge_factor=1, sigma=1, alpha=1, lmbd=1):
        # enlarge factor that indicates how larger the search region is in respect to the target region
        self.enlarge_factor = enlarge_factor
        # sigma is the variance of the gaussian in the ideal response of the filter
        self.sigma = sigma
        # alpha is the update speed, typically a low number
        self.alpha = alpha
        # lmbd is the regularization factor in the filter construction formula
        self.lmbd = lmbd


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    dataset_path = '/media/samuponz/03DD6CF02B183CCE/FRI - Ljubljana/ATCV - Advanced Topics in Computer Vision/Projects/vot2014/'  # TODO: set to the dataset path on your disk
    sequence = 'ball'  # choose the sequence you want to test
    sequence = VOTSequence(dataset_path,
                           sequence)  # using the constructor, the frames and the groundtruth positions are

    parameters = DCFParams()
    tracker = DCFTracker(parameters)

    img = cv2.imread(sequence.frame(0))  # frame is loaded in img (as a BGR image (3 channel numpy array))

    tracker.initialize(img, sequence.get_annotation(0, type='rectangle'))

    plt.figure()
    plt.imshow(tracker.h)

    img = cv2.imread(sequence.frame(1))

    _ = tracker.track(img)
    plt.figure()
    plt.imshow(tracker.h)

    plt.show()