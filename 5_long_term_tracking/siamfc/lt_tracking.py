import numpy as np
import cv2
from got10k.trackers import Tracker
from siamfc import TrackerSiamFC
from matplotlib import pyplot as plt


# UTILITY FUNCTIONS ---------------------------------------------------------------------------------------------------

def compute_max_correlation_response(patches_correlation_responses):
    # max_correlation_response is an array with the maximum values of all the correlation responses
    correlation_responses_maxima = np.amax(patches_correlation_responses, (1, 2))
    index = np.argmax(correlation_responses_maxima)
    max_correlation_response = correlation_responses_maxima[index]
    return max_correlation_response, index


# from the previous exercises utility module
def get_patch(img, center, sz):
    """
    Takes an image and returns a square containing the cropped image centered
    at the center coordinates.
    Returns also a binary mask indicating if the pixel is inside or outside the image
    """
    # takes top-left and bottom-right coordinates given the center coordinates
    x0 = round(int(center[0] - sz[0] / 2))
    y0 = round(int(center[1] - sz[1] / 2))
    x1 = int(round(x0 + sz[0]))
    y1 = int(round(y0 + sz[1]))

    # padding (necessary when the region exits the image?)
    x0_pad = max(0, -x0)
    x1_pad = max(x1 - img.shape[1] + 1, 0)
    y0_pad = max(0, -y0)
    y1_pad = max(y1 - img.shape[0] + 1, 0)

    # Crop target
    if len(img.shape) > 2:
        # crops the target in all the channels
        img_crop = img[y0 + y0_pad:y1 - y1_pad, x0 + x0_pad:x1 - x1_pad, :]
    else:
        img_crop = img[y0 + y0_pad:y1 - y1_pad, x0 + x0_pad:x1 - x1_pad]

    im_crop_padded = cv2.copyMakeBorder(img_crop, y0_pad, y1_pad, x0_pad, x1_pad, cv2.BORDER_REPLICATE)

    # crop mask tells which pixels are within the image (1) and which are outside (0)
    m_ = np.ones((img.shape[0], img.shape[1]), dtype=np.float32)
    crop_mask = m_[y0 + y0_pad:y1 - y1_pad, x0 + x0_pad:x1 - x1_pad]
    crop_mask = cv2.copyMakeBorder(crop_mask, y0_pad, y1_pad, x0_pad, x1_pad, cv2.BORDER_CONSTANT, value=0)
    return im_crop_padded, crop_mask


# LT TRACKER CLASS ----------------------------------------------------------------------------------------------------

class TrackerLT(Tracker):

    def __init__(self, net_path=None):
        super(TrackerLT, self).__init__('long term tracker')
        self.net = net_path

    def init(self, image, box):
        # In the init function the tracker is initialized:
        # The tracker is composed of st tracker + detector, these work in parallel, both have to be initialized

        # Global parameters (used by both the st and the lt tracker)
        self.confidence_th = 0.45  # 0.75 is an arbitrary value
        self.number_of_samples = 30  # 20 is an arbitrary value
        # self.failures = 0  # number of failures of the st tracker
        self.update_speed = 0.0

        # ST Tracker initialization:
        self.st_tracker = TrackerSiamFC(net_path=self.net)
        self.st_stopped = False  # at initialization, we start using the st tracker
        self.st_tracker.init(image, box)

        # Run the st tracker on the first frame in order to have the correlation response on the first frame
        # The correlation response on the first frame is used as reference
        bbox, max_correlation_response = self.st_tracker.update(image)

        self.bbox = bbox  # the resulted bbox of the st tracker is always stored in case it is needed by the detector
        # (for the gaussian sampling we need to remember the last position of the st_tracker)

        self.reference_max_correlation_response = max_correlation_response  # this it the max value of the response,
        # used to compute the confidence score

    def update(self, image):
        # The log term tracker implements both a st tracker and a detector.
        # In the previous frame, the st tracker is stopped if the confidence score is lower than the threshold.
        # At the current frame, we check the state of the tracker to decide if to run the st tracker or the detector

        # The method has to return a prediction and a score

        if self.st_stopped:
            # if the tracker has been stopped in the previous frame, the detector is run
            best_bbox, best_patch = self.detection(image)

            # store the last center of the last bbox
            last_st_center = self.st_tracker.center

            # try to track with the new position found by the detector
            self.st_tracker.change_tracker_center(best_bbox[:2])
            bbox, max_correlation_response = self.st_tracker.update(image)

            # Computation of the confidence score as the ratio between the current correlation response and the
            # reference correlation response (first frame):
            confidence_score = max_correlation_response / self.reference_max_correlation_response

            # print('Detection in new frame: confidence of the st tracker prediciton on the best detected patch:', confidence_score)

            # Stopping condition evaluation:
            if confidence_score > self.confidence_th:
                # if yes, set the new region as the one found, re-activate the st_tracker
                self.bbox = bbox
                self.st_stopped = False
                # the visual model is updated if the target is correctly detected
                self.reference_max_correlation_response = self.update_speed*max_correlation_response + \
                                                        (1-self.update_speed)*self.reference_max_correlation_response
                # in the next frame, the st_tracker will run centered at the best position found by the detector
            else:
                # if no, keep the last st_bbox and do not activate the st_tracker
                self.st_stopped = True
                # Since the target has not been detected, the last center position is restored
                self.st_tracker.change_tracker_center(last_st_center)

        else:
            # if the detector has not been stopped, we use the st_tracking in this frame
            bbox, max_correlation_response = self.st_tracker.update(image)
            # Computation of the confidence score as the ratio between the current correlation response and the
            # reference correlation response (first frame):
            confidence_score = max_correlation_response/self.reference_max_correlation_response

            # print('ST tracking: confidence score:', confidence_score)

            # Stopping condition evaluation:
            if confidence_score < self.confidence_th:
                # if the confidence is low, the st stacker is stopped
                self.st_stopped = True
                best_bbox, best_patch = self.detection(image)
                # best_bbox is returned with format [x_c, y_c, w, h]

                # store the last center of the last bbox
                # remember that st_tracker center has this format (y center, x center)
                last_st_center = self.st_tracker.center

                # try to track with the new position found by the detector
                # we get the best box center and reverse the coordinates
                best_bbox_center_yx = best_bbox[::-1][2:]
                self.st_tracker.change_tracker_center(best_bbox_center_yx)
                bbox, max_correlation_response = self.st_tracker.update(image)

                # Computation of the confidence score as the ratio between the current correlation response and the
                # reference correlation response (first frame):
                confidence_score = max_correlation_response / self.reference_max_correlation_response

                # print('Detection after st: confidence of the st tracker prediciton on the best detected patch:', confidence_score)

                # Stopping condition evaluation:
                if confidence_score > self.confidence_th:
                    # if yes, set the new region as the one found, re-activate the st_tracker
                    self.bbox = bbox
                    self.st_stopped = False
                    # the visual model is updated if the target is correctly detected
                    self.reference_max_correlation_response = self.update_speed*max_correlation_response + \
                                                        (1-self.update_speed)*self.reference_max_correlation_response
                    # in the next frame, the st_tracker will run centered at the best position found by the detector
                else:
                    # if no, keep the last st_bbox and do not activate the st_tracker
                    self.st_stopped = True
                    # Since the target has not been detected, the last center position is restored
                    self.st_tracker.change_tracker_center(last_st_center)

            else:
                # if the confidence is ok, the detector is not stopped
                self.st_stopped = False
                # the visual model is updated if the target is correctly detected
                self.reference_max_correlation_response = self.update_speed*max_correlation_response + \
                                                       (1-self.update_speed)*self.reference_max_correlation_response
                self.bbox = bbox

        return self.bbox, confidence_score

    def detection(self, image):
        # the detector is run: it samples regions in the current frame and computes the correlation response
        # on these regions.
        # If there is no region with a confidence higher than the threshold, it returns the old bbox (last st_bbox)
        # and the st_tracker is kept stopped
        # If there are regions with a confidence higher than the threshold, it returns the new bbox and st_tracker is
        # activated in the future frame.

        # generate samples
        sampling_method = 'global_uniform'
        # sampling_method = 'local_uniform'
        # sampling_method = 'gaussian'

        generated_bboxes = self.generate_random_samples(image, sampling_method)
        # generate patches
        generated_patches = self.samples_to_patches(image, generated_bboxes)
        # generate correlation responses
        patches_correlation_responses = self.st_tracker.compute_correlation_responses(generated_patches)
        # find the best patch (with max correlation response)
        max_correlation_response, index = compute_max_correlation_response(patches_correlation_responses)
        best_bbox = generated_bboxes[index]
        best_patch = generated_patches[index]
        return best_bbox, best_patch

    def generate_random_samples(self, image, sampling_method):
        # Generate samples in which perform the detection
        # The last st_tracker bbox is used to define the sample space

        H = image.shape[0]
        W = image.shape[1]

        w = self.bbox[2]
        h = self.bbox[3]

        # bbox is: [top left x, top left y, w, h]
        # we convert the bbox [x center, y center, w, h]
        x_center = self.bbox[0] + self.bbox[2] / 2
        y_center = self.bbox[1] + self.bbox[3] / 2

        if sampling_method == 'global_uniform':
            # 1) Generate samples in the whole image:

            x_coordinates = np.random.randint(np.ceil(w/2), np.ceil(W-w/2), self.number_of_samples)
            y_coordinates = np.random.randint(np.ceil(h/2), np.ceil(H-h/2), self.number_of_samples)

        elif sampling_method == 'local_uniform':
            # 2) Generate samples uniformly in the neighborhood of the target:

            enlarge_factor = 3

            x_coordinates = np.random.randint(np.ceil(x_center - enlarge_factor*w/2),
                                              np.ceil(x_center + enlarge_factor*w/2),
                                              self.number_of_samples)
            y_coordinates = np.random.randint(np.ceil(y_center - enlarge_factor*h/2),
                                              np.ceil(y_center + enlarge_factor*h/2),
                                              self.number_of_samples)

        elif sampling_method == "gaussian":
            # we generate a normal distribution centered in the last_bbox position, with a covariance matrix built with
            # the independent stds based on the size of the image
            # the output is a matrix of positions, sampled in a gaussian fashion

            # For gaussian sampling, initialize the variances of the search region
            sigma_x = self.bbox[2]
            sigma_y = self.bbox[3]

            gaussian_samples = np.random.multivariate_normal([x_center, y_center],
                                                        [[sigma_x**2, 0], [0, sigma_y**2]],
                                                        size=self.number_of_samples)
            x_coordinates = gaussian_samples[:, 0]
            y_coordinates = gaussian_samples[:, 1]

            # correction if out of image boundary
            # x_coordinates = np.where(x_coordinates > np.ceil(W - w/2), np.floor(W - w/2), x_coordinates)
            # x_coordinates = np.where(x_coordinates < np.floor(w/2), np.ceil(w/2), x_coordinates)
            # # correction if out of image boundary
            # x_coordinates = np.where(y_coordinates > np.ceil(H - h/2), np.floor(H - h/2), x_coordinates)
            # x_coordinates = np.where(y_coordinates < np.floor(H/2), np.ceil(h/2), x_coordinates)

        generated_bboxes = np.zeros(shape=(self.number_of_samples, 4))
        for i in np.arange(generated_bboxes.shape[0]):
            generated_bboxes[i, :] = [x_coordinates[i], y_coordinates[i], w, h]
        self.sampled_regions = generated_bboxes
        return generated_bboxes

    def samples_to_patches(self, image, generated_bboxes):
        # create the patches from the sampled bboxes
        sampled_patches = []
        for i in np.arange(self.number_of_samples):
            patch, _ = get_patch(image, generated_bboxes[i, :2], (generated_bboxes[i, 2], generated_bboxes[i, 3]))
            sampled_patches.append(patch)
        return sampled_patches

    # def compute_patches_correlation_responses(self, generated_patches):
    #     # generate the responses from the sampled patches
    #     correlation_responses = []
    #     for i in np.arange(self.number_of_samples):
    #         response = self.st_tracker.compute_correlation_response(generated_patches[i])
    #         correlation_responses.append(response)
    #     return correlation_responses




