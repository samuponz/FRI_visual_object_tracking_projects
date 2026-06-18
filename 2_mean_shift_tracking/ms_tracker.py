import math

import cv2

from ex2_utils import Tracker, create_epanechnik_kernel, get_patch, extract_histogram, backproject_histogram
import numpy as np


# MEAN-SHIFT TRACKER --------------------------------------------------------------------------------
# MeanShiftTracker inherits from the Tracker class
from sequence_utils import VOTSequence


class MeanShiftTracker(Tracker):

    def initialize(self, image, region):
        """
        Initialization of the tracker: it takes the initialization frame and the region (coordinates of a bbox)
        where the target is located. Given the region, it crops the image in that region and compute the visual model,
        i.e. the wighted sub-sampled histogram of the cropped image.
        It updates the values of the attributes (region_size, target_position, kernel, q of the objects) that will be
        used as starting values for the the track method.
        :param image: initial frame
        :param region: tl coordinates and size of the bbox containing the target
        :return: nothing, but it sets the values of some attributes
        """

        # NOTE: the get_annotation function in sequence_utils always ensures that the region is a square, thus a list
        # of 4 values.

        # Get region size (make sure the region is odd sized)
        self.region_size = (int((np.ceil(region[2]//2)*2 + 1)*self.parameters.enlarge_factor),
                            int((np.ceil(region[3]//2)*2 + 1)*self.parameters.enlarge_factor))

        # (region_size is stored in a tuple so it cannot be modified)

        # NOTE: THE REGION COORDINATES ARE STORED WITH CARTESIAN INDEXING (x, y), a region is [x, y, w, h]
        # We are working with images(matrices) that uses MATRIX INDEXING (i-row, j-col)
        # therefore the width of the region goes over the cols, the height goes over the rows

        # Find the central position of the region (center of the target)
        # CHECK THAT THE ROUNDING HERE IS THE SAME IN THE TRACK METHOD
        self.target_position = (round(region[0] + region[2]/2), round(region[1] + region[3]/2))

        # NOTE: the patch and the kernel need to have size (height, width), not (with, height)!

        # Compute the Epanechnikov kernel with the region size -> obtain weights
        self.kernel = create_epanechnik_kernel(self.region_size[1], self.region_size[0], self.parameters.sigma)

        # Crop the image in that region to get the patch
        search_region, _ = get_patch(image, self.target_position, self.kernel.shape)

        # # Multiply the image by the kernel
        # search_region_float = search_region.astype("float")
        # for channel in range(search_region.shape[2]):
        #     search_region_float[:, :, channel] *= np.transpose(self.kernel)
        # modulated_search_region = search_region_float.astype("uint8")  # from float to values in [0, 255]

        # Compute the weighted histogram q
        # self.q = extract_histogram(modulated_search_region, n_bins=self.parameters.n_bins)
        self.q = extract_histogram(search_region, n_bins=self.parameters.n_bins, weights=self.kernel)
        self.q = np.divide(self.q, np.sum(self.q))
        # Normalize the histogram if it is not already done it in the function
        # q is already normalized because the kernel is normalized inside its build function

    def track(self, image):
        """
        Tracking method. Given a frame of a video and given the position of the target in the previous frame, it
        computes the new position of the target in this frame. To estimate the new position the method runs the mean
        shift algorithm on the result of the backprojection of a histogram of wights (v), computed as the square root of
        the ratio between the histogram of the target and the histogram of the current search region.

        :param image: current frame in which ww want to estimate the position of the target
        :return: the estimated region of the target: top-left coordinates and size
        """

        # parameters
        custom_eps = 1e-3  # just a small value greater zero to prevent true_division errors

        # INIT of MS algorithm ----------------------------------------------------------------------------------------

        # Init iteration counter:
        counter = 0

        # Initialize the ms_vector that will be updated at every iteration
        ms_vect = np.empty(len(self.target_position), dtype=float)

        # First estimation of the target position = last frame target position
        x_k = np.array(self.target_position, dtype=float)

        # Initial shift (initialized at something greater than tol just to enter the while loop)
        ms_vect_norm = self.parameters.tol + 1

        # LOOP of MS algorithm ----------------------------------------------------------------------------------------
        while ms_vect_norm > self.parameters.tol and counter < self.parameters.n_max:

            # get the patch centered at the estimated position, with diameter equal to the bandwidth
            search_region, _ = get_patch(image, self.target_position, self.kernel.shape)

            # Creation of the matrices of coordinates centered in the patch: x_i, y_i
            w = search_region.shape[1]
            h = search_region.shape[0]
            w2 = int(np.floor(w / 2))
            h2 = int(np.floor(h / 2))
            x_i, y_i = np.meshgrid(np.arange(-w2, w2 + 1), np.arange(-h2, h2 + 1))

            # # Multiply the image by the kernel
            # search_region_float = search_region.astype("float")
            # for channel in range(search_region.shape[2]):
            #     search_region_float[:, :, channel] *= np.transpose(self.kernel)
            # modulated_search_region = search_region_float.astype("uint8")  # from float to values in [0, 255]

            # n bins histogram of the search region
            # p = extract_histogram(modulated_search_region, self.parameters.n_bins)
            p = extract_histogram(search_region, self.parameters.n_bins, weights=self.kernel)
            p = np.divide(p, np.sum(p))

            # Ratio of the two histograms
            v = np.sqrt(np.divide(self.q, p + custom_eps))
            v = np.divide(v, np.sum(v))

            # image backprojected from the v histogram
            weight_distribution = backproject_histogram(search_region, v, self.parameters.n_bins)

            # check if all the weights are zero
            if np.sum(weight_distribution) == 0:
                ms_vect[0] = 0
                ms_vect[0] = 0
            else:
                # x_new is the new estimation of the shift towards the mode of the pdf
                ms_vect[0] = np.sum(weight_distribution * x_i) / (np.sum(weight_distribution))
                ms_vect[1] = np.sum(weight_distribution * y_i) / (np.sum(weight_distribution))

            # The norm of the ms_vect, it should progressively decrease ad reach zero
            ms_vect_norm = np.linalg.norm(ms_vect, 2)

            x_k[0] = x_k[0] + ms_vect[0]  # Update the estimation of the mode
            x_k[1] = x_k[1] + ms_vect[1]  # Update the estimation of the mode
            counter += 1

            # Storing x_k and err in the vectors containing the history of these values
            # mode_pos = np.concatenate((mode_pos, x_k.reshape(1, 2)), axis=0)
            # steps = np.append(steps, ms_vect)

        # Update estimated target position
        self.target_position = (x_k[0], x_k[1])

        # Update visual model (q histogram of the target)
        self.q = (1 - self.parameters.alpha)*self.q + self.parameters.alpha*p

        # Returns the coordinates of the new estimated region
        return [self.target_position[0] - self.region_size[0] / 2.0,
                self.target_position[1] - self.region_size[1] / 2.0,
                self.region_size[0],
                self.region_size[1]]

        # return [self.target_position[0] - self.region_size[0]/2.0,
        #         self.target_position[1] - self.region_size[1]/2.0,
        #         self.region_size[0],
        #         self.region_size[1]], p, v


# class of the parameters of the tracker, here we generate all the parameters with a custom or a default value.
# Remember that different constructors can be created just by changing the number of parameters for every constructor!
class MSParams():
    def __init__(self, sigma=2, n_bins=4, alpha=0, tol=0.001, n_max=20, enlarge_factor=1.25):

        # Search region parameters
        self.enlarge_factor = enlarge_factor  # How the search region is bigger than the target region
        # This can be seen as a ratio between the size of the search region and the one of the target region

        # Visual model parameters
        self.n_bins = n_bins  # number of bins in the sub-sampled histogram
        self.sigma = sigma  # parameter of the Epanechnikov kernel
        self.alpha = alpha  # model update parameter

        # MS algorithm convergence parameters
        self.tol = tol  # threshold on the L2 norm of the MS vector
        self.n_max = n_max  # imposed max number of iterations


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    dataset_path = '/media/samuponz/03DD6CF02B183CCE/FRI - Ljubljana/ATCV - Advanced Topics in Computer Vision/Projects/Project 2/vot2014/'  # TODO: set to the dataset path on your disk
    sequence = 'basketball'  # choose the sequence you want to test
    sequence = VOTSequence(dataset_path, sequence)  # using the constructor, the frames and the groundtruth positions are

    parameters = MSParams()
    tracker = MeanShiftTracker(parameters)

    img = cv2.imread(sequence.frame(0))  # frame is loaded in img (as a BGR image (3 channel numpy array))

    tracker.initialize(img, sequence.get_annotation(0, type='rectangle'))

    plt.figure()
    plt.plot(tracker.q)

    img = cv2.imread(sequence.frame(1))

    _, p, v = tracker.track(img)

    plt.figure()
    plt.plot(p)
    plt.figure()
    plt.plot(v)

    # cv2.imshow('full frame', img)
    # cv2.waitKey()

    plt.show()

