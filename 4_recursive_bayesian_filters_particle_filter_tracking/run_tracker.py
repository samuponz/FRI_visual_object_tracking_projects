import time

import cv2
import numpy as np
from matplotlib import pyplot as plt

from pf_tracker_ncv_N_100 import PFParams_ncv_N_100, PFTracker_ncv_N_100
from sequence_utils import VOTSequence

# from ncc_tracker_example import NCCTracker, NCCParams
# from ms_tracker import MeanShiftTracker, MSParams


def test_tracker(dataset_path, sequence_name, tracker_parameters, setup_parameters):
    # create VOT-sequence object
    sequence = VOTSequence(dataset_path,
                           sequence_name)  # using the constructor, the frames and the groundtruth positions are
    # loaded into the relative attributes of the object

    init_frame = 0
    n_failures = 0

    # Create parameters and tracker objects:

    # parameters = NCCParams()  # We create a class to menage the parameters of the tracker, we create the parameters
    # tracker = NCCTracker(parameters)  # We instantiate the tracker passing the parameters to the class constructor

    # Create parameters and tracker objects
    parameters = PFParams_ncv_N_100(**tracker_parameters)
    tracker = PFTracker_ncv_N_100(parameters)

    # Counter
    time_all = 0

    # Initialize visualization window
    if setup_parameters['show_sequence']:
        sequence.initialize_window(setup_parameters['win_name'])

    # tracking loop - goes over all frames in the video sequence

    frame_idx = 0  # starts from the first frame in the sequence

    while frame_idx < sequence.length():  # until the last frame

        img = cv2.imread(sequence.frame(frame_idx))  # frame is loaded in img (as a BGR image (3 channel numpy array))

        # initialize or track
        if frame_idx == init_frame:
            # initialize tracker (at the beginning of the sequence or after tracking failure)
            t_ = time.time()
            tracker.initialize(img, sequence.get_annotation(frame_idx, type='rectangle'))
            time_all += time.time() - t_
            predicted_bbox = sequence.get_annotation(frame_idx, type='rectangle')  # the predicted box in the
            # initialization is the groundtruth bounding box, this is why we use the again get_annotation (that uses
            # the groundtruth pos)
        else:
            # track on current frame - predict bounding box
            t_ = time.time()
            predicted_bbox = tracker.track(img)
            time_all += time.time() - t_

        # compute overlap (needed to determine failure of a tracker)
        gt_bb = sequence.get_annotation(frame_idx, type='rectangle')
        o = sequence.overlap(predicted_bbox, gt_bb)  # this computes the bounding boxes areas and computes IoU

        # draw ground-truth and predicted bounding boxes, frame numbers and show image
        if setup_parameters['show_sequence']:
            if setup_parameters['show_gt']:
                sequence.draw_region(img, gt_bb, (0, 255, 0), 1)
            sequence.draw_region(img, predicted_bbox, (0, 0, 255), 2)
            sequence.draw_text(img, '%d/%d' % (frame_idx + 1, sequence.length()), (25, 25))
            sequence.draw_text(img, 'Fails: %d' % n_failures, (25, 55))
            sequence.show_image(img, setup_parameters['video_delay'])

        if o > 0 or not setup_parameters['reinitialize']:
            # increase frame counter by 1
            frame_idx += 1
        else:
            # increase frame counter by 5 and set re-initialization to the next frame
            frame_idx += 5
            init_frame = frame_idx
            n_failures += 1

    tracking_speed = sequence.length() / time_all

    print('VOT sequence: ' + sequence_name)
    print('Tracking speed: %.1f FPS' % tracking_speed)
    print('Tracker failed %d times' % n_failures)

    return tracking_speed, n_failures


def multiple_tests(param, param_list, dataset_path, VOT14_sequences, sequence_name, ms_tracker_parameters,
                   setup_parameters):

    temp_ms_tracker_parameters = ms_tracker_parameters.copy()
    # this is done in order to not modify the original dictionary

    failures = np.zeros(len(param_list), dtype=int)
    for i in np.arange(len(param_list)):
        temp_ms_tracker_parameters[param] = param_list[i]
        tracking_speed, n_failures = test_tracker(dataset_path, sequence_name, temp_ms_tracker_parameters,
                                                  setup_parameters)
        failures[i] = n_failures
    return failures


def plot_failures(param, param_values, failures):
    plt.figure()
    plt.plot(param_values, failures)
    plt.title('Performance w.r.t. ' + param)
    plt.xlabel(param)
    plt.ylabel('Number of failures')
    plt.grid(True)


if __name__ == '__main__':
    # DATASET PARAMETERS --------------------------------------------------

    # Set the path to directory where you have the sequences
    dataset_path = \
        '/media/samuponz/03DD6CF02B183CCE/FRI - Ljubljana/ATCV - ' \
        'Advanced Topics in Computer Vision/Projects/vot2014/'

    VOT14_sequences = ['ball', 'basketball', 'bicycle', 'bolt', 'car', 'david', 'diving', 'drunk', 'fernando', 'fish1',
                       'fish2', 'gymnastics', 'hand1', 'hand2', 'jogging', 'motocross', 'polarbear', 'skating', 'sphere',
                       'sunshade', 'surfing', 'torus', 'trellis', 'tunnel', 'woman']

    # TESTING PARAMETERS -------------------------------------------------

    # standard tracker parameters
    # ms_tracker_parameters1 = {
    #     'sigma': 2.5,
    #     'n_bins': 8,
    #     'alpha': 0,
    #     'tol': 0.1,
    #     'n_max': 10,
    #     'enlarge_factor': 1.25}
    #
    # ms_tracker_parameters2 = {
    #     'sigma': 2.5,
    #     'n_bins': 16,
    #     'alpha': 0.05,
    #     'tol': 0.95,
    #     'n_max': 8,
    #     'enlarge_factor': 1.25}

    # dcf_tracker_parameters1 = {
    #     'enlarge_factor': 1.5,
    #     'sigma': 2,
    #     'alpha': 0.1,
    #     'lmbd': 0.01,
    #     }

    pf_tracker_parameters = {
        'motion_model': 'nca',
        'q': 1,
        'kernel_param': 1,
        'n_bins': 8,
        'alpha': 0.05,
        'n_particles': 100,
    }

    setup_parameters = {
        'win_name': 'Tracking window',
        'reinitialize': True,
        'show_sequence': False,
        'show_gt': True,  # Flag for drawing the groundtruth bounding box
        'video_delay': 1
        ,
        'font': cv2.FONT_HERSHEY_PLAIN}

    # TESTING ----------------------------------------------------------------------

    # Test the algorithm with standard parameters on 1 VOT sequence
    # name = 'david'
    # test_tracker(dataset_path, name, pf_tracker_parameters, setup_parameters)

    # Test the algorithm with standard parameters on all the VOT sequences
    # for i in np.arange(len(VOT14_sequences)):
    #     test_tracker(dataset_path, VOT14_sequences[i], pf_tracker_parameters, setup_parameters)

    # PF tracker -----------------------------------------------------------------------------------------------------
    sequence_name = 'bolt'
    q_list = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75]

    q_failures = multiple_tests('q', q_list, dataset_path, VOT14_sequences, sequence_name,
                                    pf_tracker_parameters, setup_parameters)

    plot_failures('q', q_list, q_failures)
    plt.show()

    # DCF tracker -----------------------------------------------------------------------------------------------------
    # sequence_name = 'basketball'
    # sigma_list = [1, 2, 3, 4, 5]
    # ef_list = [1, 1.25, 1.3, 1.5, 1.75, 2]
    # alpha_list = np.arange(0.1, 1, 0.05)
    # lmbd_list = np.arange(0, 0.1, 0.01)
    #
    # sigma_failures = multiple_tests('sigma', sigma_list, dataset_path, VOT14_sequences, sequence_name,
    #                                  dcf_tracker_parameters1, setup_parameters)
    #
    # ef_failures = multiple_tests('enlarge_factor', ef_list, dataset_path, VOT14_sequences, sequence_name,
    #                               dcf_tracker_parameters1, setup_parameters)
    #
    # alpha_failures = multiple_tests('alpha', alpha_list, dataset_path, VOT14_sequences, sequence_name,
    #                                 dcf_tracker_parameters1, setup_parameters)
    #
    # lmbd_failures = multiple_tests('lmbd', lmbd_list, dataset_path, VOT14_sequences, sequence_name,
    #                                 dcf_tracker_parameters1, setup_parameters)
    #
    # plot_failures('sigma', sigma_list, sigma_failures)
    # plot_failures('enlarge_factor', ef_list, ef_failures)
    # plot_failures('alpha', alpha_list, alpha_failures)
    # plot_failures('lmbd', lmbd_list, lmbd_failures)
    # plt.show()

    # MS tracker -----------------------------------------------------------------------------------------------------
    # Test on one single VOT sequence with different parameters
    # sequence_name = 'jogging'
    # n_bins_list = [4, 8, 16, 32, 64]
    # tol_list = np.linspace(0.001, 1, 20)
    # alpha_list = np.arange(0, 1, 0.05)
    # n_max_list = np.arange(5, 20, 1, dtype=int)
    #
    # n_bins_failures = multiple_tests('n_bins', n_bins_list, dataset_path, VOT14_sequences, sequence_name,
    #                                  ms_tracker_parameters1, setup_parameters)
    #
    # tol_failures = multiple_tests('tol', tol_list, dataset_path, VOT14_sequences, sequence_name,
    #                               ms_tracker_parameters1, setup_parameters)
    #
    # alpha_failures = multiple_tests('alpha', alpha_list, dataset_path, VOT14_sequences, sequence_name,
    #                                 ms_tracker_parameters1, setup_parameters)
    #
    # n_max_failures = multiple_tests('n_max', n_max_list, dataset_path, VOT14_sequences, sequence_name,
    #                                 ms_tracker_parameters1, setup_parameters)
    #
    # plot_failures('n_bins', n_bins_list, n_bins_failures)
    # plot_failures('tol', tol_list, tol_failures)
    # plot_failures('alpha', alpha_list, alpha_failures)
    # plot_failures('n_max', n_max_list, n_max_failures)
    # plt.show()
