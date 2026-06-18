import argparse
import os
import cv2

from tools.sequence_utils import VOTSequence
from tools.sequence_utils import save_results

from siamfc import TrackerSiamFC
from siamfc.lt_tracking import TrackerLT  # not sure about this


def evaluate_tracker(dataset_path, network_path, results_dir, visualize):

    # Create a list of the sequences to analyze
    sequences = []
    with open(os.path.join(dataset_path, 'list.txt'), 'r') as f:
        for line in f.readlines():
            sequences.append(line.strip())

    # Create a TrackerSiamFC object
    # tracker = TrackerSiamFC(net_path=network_path)
    tracker = TrackerLT(net_path=network_path)

    for sequence_name in sequences:
        
        print('Processing sequence:', sequence_name)

        bboxes_path = os.path.join(results_dir, '%s_bboxes.txt' % sequence_name)
        scores_path = os.path.join(results_dir, '%s_scores.txt' % sequence_name)

        if os.path.exists(bboxes_path) and os.path.exists(scores_path):
            print('Results on this sequence already exists. Skipping.')
            continue

        # Using the sequence.utils for testing the tracker
        sequence = VOTSequence(dataset_path, sequence_name)

        img = cv2.imread(sequence.frame(0))
        gt_rect = sequence.get_annotation(0)
        tracker.init(img, gt_rect)
        results = [gt_rect]
        scores = [[10000]]  # a very large number - very confident at initialization

        if visualize:
            cv2.namedWindow('win', cv2.WINDOW_AUTOSIZE)
        for i in range(1, sequence.length()):

            # if i % 100 == 0:
            #     print('Processing sequence', sequence_name, ', frame:', i)
            print('Processing sequence', sequence_name, ', frame:', i)
            img = cv2.imread(sequence.frame(i))
            prediction, score = tracker.update(img)
            results.append(prediction)
            scores.append([score])

            if visualize:
                tl_ = (int(round(prediction[0])), int(round(prediction[1])))
                br_ = (int(round(prediction[0] + prediction[2])), int(round(prediction[1] + prediction[3])))
                if score > tracker.confidence_th:
                    cv2.rectangle(img, tl_, br_, (0, 0, 255), 1)
                if tracker.st_stopped:
                    for i in range(tracker.sampled_regions.shape[0]):
                        # circles:
                        # cv2.circle(img, tuple(tracker.sampled_regions[i, :][0:2].astype(int)), 5, (0, 255, 0))
                        # bboxes:
                        # format: [x_coordinates[i], y_coordinates[i], w, h]
                        tl_ = (int(tracker.sampled_regions[i, 0] - tracker.sampled_regions[i, 2]//2),
                               int(tracker.sampled_regions[i, 1] - tracker.sampled_regions[i, 3]//2))
                        br_ = (int(tracker.sampled_regions[i, 0] + tracker.sampled_regions[i, 2]//2),
                               int(tracker.sampled_regions[i, 1] + tracker.sampled_regions[i, 3]//2))
                        cv2.rectangle(img, tl_, br_, (0, 255, 0), 1)

                cv2.imshow('win', img)
                key_ = cv2.waitKey(0)
                if key_ == 27:
                    exit(0)
        
        save_results(results, bboxes_path)
        save_results(scores, scores_path)


# Running the tracker from console, take the parameters in the command line
# parser = argparse.ArgumentParser(description='SiamFC Runner Script')
# parser.add_argument("--dataset", help="Path to the dataset", required=True, action='store')
# parser.add_argument("--net", help="Path to the pre-trained network", required=True, action='store')
# parser.add_argument("--results_dir", help="Path to the directory to store the results", required=True, action='store')
# parser.add_argument("--visualize", help="Show ground-truth annotations", required=False, action='store_true')
# args = parser.parse_args()

# Running the tracker from IDE, take the parameters written here
dataset_path = '/home/samuponz/PycharmProjects/ATCV/SiamFC/dataset_lt'
net_path = '/home/samuponz/PycharmProjects/ATCV/SiamFC/siamfc_net.pth'
results_dir = '/home/samuponz/PycharmProjects/ATCV/SiamFC/results'
visualize = True

# evaluate_tracker(args.dataset, args.net, args.results_dir, args.visualize)
evaluate_tracker(dataset_path, net_path, results_dir, visualize)
