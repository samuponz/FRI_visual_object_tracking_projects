import math
import numpy as np
import cv2
import re
import glob
from matplotlib import pyplot as plt


# My functions

def build_video(directory):
    img_array = []
    numbers = re.compile(r'(\d+)')

    def numericalSort(value):
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts

    for filename in sorted(glob.glob('camera_movement/*.jpg'), key=numericalSort):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter('camera_movement.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


def show_images(titles, images):
    for i in range(len(titles)):
        plt.figure()
        if len(images[i].shape) == 2:
            plt.imshow(images[i], cmap="gray", interpolation='bicubic', vmin=0, vmax=255, origin='upper')
        else:
            plt.imshow(images[i], interpolation='bicubic', vmin=0, vmax=255, origin='upper')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

    plt.show()


def read_images(prev, succ):
    im1 = cv2.imread(prev, cv2.IMREAD_GRAYSCALE)
    im2 = cv2.imread(succ, cv2.IMREAD_GRAYSCALE)
    return im1, im2


def synthetic_images_generation():
    im1 = np.random.rand(200, 200).astype(np.float32)
    im2 = im1.copy()
    im2 = rotate_image(im2, -1)
    return im1, im2


def apply_mask(img, mask):
    if len(img.shape) == 2:
        img = np.stack((img, img, img), axis=2)  # convert from grayscale to rgb
    for i in np.arange(img.shape[2]):  # for every channel apply the mask
        img[:, :, i] = np.where(mask[:, :, i] == 255, 255, img[:, :, i])
    return img


# Given functions

def create_cosine_window(target_size):
    # target size is in the format: (width, height)
    # output is a matrix of dimensions: (width, height)
    return cv2.createHanningWindow((target_size[0], target_size[1]), cv2.CV_32F)


def create_gauss_peak(target_size, sigma):
    # target size is in the format: (width, height)
    # sigma: parameter (float) of the Gaussian function
    # note that sigma should be small so that the function is in a shape of a peak
    # values that make sens are approximately from the interval: ~(0.5, 5)
    # output is a matrix of dimensions: (width, height)
    w2 = math.floor(target_size[0] / 2)
    h2 = math.floor(target_size[1] / 2)
    [X, Y] = np.meshgrid(np.arange(-w2, w2 + 1), np.arange(-h2, h2 + 1))
    G = np.exp(-X**2 / (2 * sigma**2) - Y**2 / (2 * sigma**2))
    G = np.roll(G, (-h2, -w2), (0, 1))
    return G


def gaussderiv(img, sigma):
    # sigma is the std of the gaussian filter use prior the application of the filter
    x = np.array(list(range(math.floor(-3.0 * sigma + 0.5), math.floor(3.0 * sigma + 0.5) + 1)))
    G = np.exp(-x ** 2 / (2 * sigma ** 2))
    G = G / np.sum(G)

    D = -2 * (x * np.exp(-x ** 2 / (2 * sigma ** 2))) / (np.sqrt(2 * math.pi) * sigma ** 3)
    D = D / (np.sum(np.abs(D)) / 2)

    Dx = cv2.sepFilter2D(img, -1, D, G)
    Dy = cv2.sepFilter2D(img, -1, G, D)

    return Dx, Dy


def gaussmooth(img, sigma):
    x = np.array(list(range(math.floor(-3.0 * sigma + 0.5), math.floor(3.0 * sigma + 0.5) + 1)))
    G = np.exp(-x ** 2 / (2 * sigma ** 2))
    G = G / np.sum(G)
    return cv2.sepFilter2D(img, -1, G, G)


def rotate_image(img, angle):
    center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    return rotated


def generate_responses_1():
    """
    Generates the function where to test the mean shift implementation
    """
    responses = np.zeros((100, 100), dtype=np.float32)
    responses[70, 50] = 1
    responses[50, 70] = 0.5
    return gaussmooth(responses, 10)


def generate_responses_2():
    """
    Generates the function where to test the mean shift implementation
    """
    responses = np.zeros((100, 100), dtype=np.float32)
    responses[60, 50] = 1
    responses[60, 30] = 0.8
    responses[45, 40] = 0.5
    return gaussmooth(responses, 5)


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


def create_epanechnik_kernel(width, height, sigma):
    """
    creates the epanechnik kernel of a given size and with a given smoothing coefficient
    """
    # make sure that width and height are odd

    w2 = int(math.floor(width / 2))
    h2 = int(math.floor(height / 2))

    [X, Y] = np.meshgrid(np.arange(-w2, w2 + 1), np.arange(-h2, h2 + 1))

    # normalization of the kernel
    X = X / np.max(X)
    Y = Y / np.max(Y)

    kernel = (1 - ((X / sigma)**2 + (Y / sigma)**2))
    kernel = kernel / np.max(kernel)
    kernel[kernel < 0] = 0
    return kernel


def extract_histogram(patch, n_bins, weights=None):
    """
    Extracts the histogram, already sub-sampled to n_bins levels!
    """
    # Note: input patch must be a BGR image (3 channel numpy array)

    # Computes an image the same size of the patch, containing in evey position the index of the intensity level in
    # which the intensity of that pixel resides. It is not properly the sub_quantized image, for that it would be
    # necessary to map every channel index to the respective channel value!.
    channel_bin_idxs = np.floor((patch.astype(np.float32) / float(255)) * float(n_bins - 1))

    # Converts the 3D to a 2D
    bin_idxs = (channel_bin_idxs[:, :, 0] * n_bins ** 2 + channel_bin_idxs[:, :, 1] * n_bins + channel_bin_idxs[:, :, 2]).astype(np.int32)

    # count bin indices to create histogram (use per-pixel weights if given)
    if weights is not None:
        histogram_ = np.bincount(bin_idxs.flatten(), weights=weights.flatten())
    else:
        histogram_ = np.bincount(bin_idxs.flatten())
    # zero-pad histogram (needed since bin_count function does not generate histogram with n_bins**3 elements)
    histogram = np.zeros((n_bins ** 3, 1), dtype=histogram_.dtype).flatten()
    histogram[:histogram_.size] = histogram_
    return histogram


def backproject_histogram(patch, histogram, n_bins):
    # Note: input patch must be a BGR image (3 channel numpy array)
    # convert each pixel intensity to the one of n_bins bins
    channel_bin_idxs = np.floor((patch.astype(np.float32) / float(255)) * float(n_bins - 1))
    # calculate bin index of a 3D histogram
    bin_idxs = (channel_bin_idxs[:, :, 0] * n_bins ** 2 + channel_bin_idxs[:, :, 1] * n_bins + channel_bin_idxs[:, :, 2]).astype(np.int32)

    # use histogram us a lookup table for pixel backprojection
    backprojection = np.reshape(histogram[bin_idxs.flatten()], (patch.shape[0], patch.shape[1]))
    return backprojection


# base class for tracker
class Tracker():

    # base constructor assigns the values of the custom parameters
    def __init__(self, params):
        self.parameters = params

    # initialization method of the tracker needs a frame and a region where to start searching.
    def initialize(self, image, region):
        raise NotImplementedError

    # tracking method only needs the new frame where to search
    def track(self, image):
        raise NotImplementedError