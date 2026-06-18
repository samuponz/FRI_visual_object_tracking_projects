from ex1_utils import gaussmooth, gaussderiv, apply_mask

import math
import numpy as np
import cv2

from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb


# OF visualization

def show_lk_results(im1, im2, u_lk, v_lk, mask_lk):
    """
    Plot the LK results of the tests done on the given pair of images
    """
    fig1, ((ax1_11, ax1_12), (ax1_21, ax1_22)) = plt.subplots(2, 2)
    ax1_11.imshow(im1, cmap="gray", interpolation='bicubic', origin='upper')
    ax1_12.imshow(im2, cmap="gray", interpolation='bicubic', origin='upper')
    show_flow(u_lk, v_lk, ax1_21, type='angle')
    show_flow(u_lk, v_lk, ax1_22, type='field', set_aspect=True)
    fig1.suptitle('Lucas-Kanade Optical Flow')

    fig2, ((ax2_11, ax2_12), (ax2_21, ax2_22)) = plt.subplots(2, 2)
    ax2_11.imshow(im1, cmap="gray", interpolation='bicubic', origin='upper')
    ax2_12.imshow(mask_lk, interpolation='bicubic', origin='upper')
    ax2_21.imshow(apply_mask(im1, mask_lk), interpolation='bicubic', origin='upper')
    show_flow(u_lk, v_lk, ax2_22, type='angle')
    fig2.suptitle('Lucas-Kanade Reliability')


def show_hs_results(im1, im2, u_hs, v_hs):
    """
    Plot the HS results of the tests done on the given pair of images
    """
    fig1, ((ax1_11, ax1_12), (ax1_21, ax1_22)) = plt.subplots(2, 2)
    ax1_11.imshow(im1, cmap="gray", interpolation='bicubic', origin='upper')
    ax1_12.imshow(im2, cmap="gray", interpolation='bicubic', origin='upper')
    show_flow(u_hs, v_hs, ax1_21, type='angle')
    show_flow(u_hs, v_hs, ax1_22, type='field', set_aspect=True)
    fig1.suptitle('Horn-Schunck Optical Flow')


def show_both_results(im1, im2, u_lk, v_lk, u_hs, v_hs):
    """
    Plot the HS results of the tests done on the given pair of images
    """
    fig1, ((ax1_11, ax1_12), (ax1_21, ax1_22), (ax1_31, ax1_32)) = plt.subplots(3, 2)
    ax1_11.imshow(im1, cmap="gray", interpolation='bicubic', origin='upper')
    ax1_12.imshow(im2, cmap="gray", interpolation='bicubic', origin='upper')
    show_flow(u_lk, v_lk, ax1_21, type='angle')
    show_flow(u_lk, v_lk, ax1_22, type='field', set_aspect=True)
    show_flow(u_hs, v_hs, ax1_31, type='angle')
    show_flow(u_hs, v_hs, ax1_32, type='field', set_aspect=True)
    fig1.suptitle('Optical Flow on real images')


def show_flow(U, V, ax, type='field', set_aspect=False):
    """
    U, V: optical flow components
    ax: axis
    type: field (intuitive but not good), magnitude, angle-magnitude (correct representation)
    """
    if type == 'field':
        scaling = 0.1
        u = cv2.resize(gaussmooth(U, 1.5), (0, 0), fx=scaling, fy=scaling)
        v = cv2.resize(gaussmooth(V, 1.5), (0, 0), fx=scaling, fy=scaling)

        x_ = (np.array(list(range(1, u.shape[1] + 1))) - 0.5) / scaling
        y_ = -(np.array(list(range(1, u.shape[0] + 1))) - 0.5) / scaling
        x, y = np.meshgrid(x_, y_)

        ax.quiver(x, y, -u * 5, v * 5)
        if set_aspect:
            ax.set_aspect(1.)
    elif type == 'magnitude':
        magnitude = np.sqrt(U ** 2 + V ** 2)
        ax.imshow(np.minimum(1, magnitude), origin='upper')
    elif type == 'angle':
        angle = np.arctan2(V, U) + math.pi
        im_hsv = np.concatenate((np.expand_dims(angle / (2 * math.pi), -1),
                                 np.expand_dims(np.ones(angle.shape, dtype=np.float32), -1),
                                 np.expand_dims(np.ones(angle.shape, dtype=np.float32), -1)), axis=-1)
        ax.imshow(hsv_to_rgb(im_hsv), origin='upper')
    elif type == 'angle_magnitude':
        magnitude = np.sqrt(U ** 2 + V ** 2)
        angle = np.arctan2(V, U) + math.pi
        im_hsv = np.concatenate((np.expand_dims(angle / (2 * math.pi), -1),
                                 np.expand_dims(np.minimum(1, magnitude), -1),
                                 np.expand_dims(np.ones(angle.shape, dtype=np.float32), -1)), axis=-1)
        ax.imshow(hsv_to_rgb(im_hsv), origin='upper')
    else:
        print('Error: unknown optical flow visualization type.')
        exit(-1)


# OF computation

def lk_reliability(sumxx, sumyy, D, cond_th, low_th):
    """
    Check if the Lucas-Kanade method is reliable. The eigenvalues of the covariance matrices of the neighborhoods are
    computed and two conditions are verified: eigs not too small, condition number not too high. If one of these
    conditions is not verified, the OF vector for that neighborhood is flagged as not reliable. A mask of the not
    reliable OF vectors in the image is returned.
    :param sumxx: factor sum Ix*Ix over neighbors
    :param sumyy: factor sum Ix*Ix over neighbors
    :param D: determinants of the covariance matrices of the neighborhoods
    :param low_th:
    :param cond_th:
    :return mask: boolean image with the not reliable neighborhoods (and therefore the vectors for these neighbors)
    """

    # Eigenvalues computation: solve lmbd^2 - (sumxx+sumyy)*lmbd + D = 0
    a = 1
    b = -(sumxx+sumyy)
    c = D
    delta = b**2 - 4*a*c

    # solutions:
    lmbd1 = (-b+np.sqrt(delta))/(2*a)
    lmbd2 = (-b-np.sqrt(delta))/(2*a)

    # Tensor with all the eigenvalues not needed):
    # eigs = np.stack([lmbd1, lmbd2],axis=2)

    # It is correct to obtain values of cond = inf, since some pairs of eigs are correctly equal to 0.
    # I will just change the values of lmbd2 from zero to eps (2.22e-16), so that I don't get true division errors.
    # Trivial solution: substitute these zeros with the machine precision of the used data.type
    lmbd2_eps = lmbd2.copy()
    lmbd2_eps[lmbd2_eps == 0] = np.finfo(lmbd2_eps.dtype).eps  # This solves division by zeros
    # All 0.000e+00 -> 2.220e-16 (eps for float64)
    # Computation of the condition number of the matrix
    # lmbd1 > lmbd2, always. Check with: mask = np.greater_equal(lmbd1,lmbd2)
    cond = np.divide(lmbd1, lmbd2_eps)

    # Condition number mask
    mask_cond = cond >= cond_th  # check if condition number is higher than the threshold
    # Low valued eigenvalues mask
    mask_low = (lmbd1 <= low_th) & (lmbd2 <= low_th)  # check if both eigs are low valued
    # Final mask
    mask = mask_low | mask_cond  # flag if at least one of the conditions is true
    # Converting the mask to a rgb (red) image
    mask = mask.astype(np.uint8)  # convert to an unsigned byte
    mask *= 255
    mask = np.stack((mask, np.zeros(mask.shape), np.zeros(mask.shape)), axis=2)
    return mask


def lucaskanade(im1, im2, n, cond_th, low_th):
    """
    im1 - first image matrix (grayscale)
    im2 - second image matrix (grayscale)
    n - size of the neighborhood (N x N)
    """

    # Compute Ix, Iy, It:
    It = gaussmooth(im2 - im1, sigma=1)  # temporal derivative
    # Ix, Iy are computed by convolution. Use the gaussderiv to compute them (one frame needed)
    Ix1, Iy1 = gaussderiv(im1, sigma=1)  # horizontal spatial derivative
    Ix2, Iy2 = gaussderiv(im2, sigma=1)  # vertical spatial derivative
    Ix = 0.5*(Ix1+Ix2)  # horizontal horizontal spatial derivative
    Iy = 0.5*(Iy1+Iy2)  # vertical horizontal spatial derivative

    # Compute the local sums with the convolution
    sumxx = cv2.filter2D(src=np.multiply(Ix, Ix), ddepth=-1, kernel=np.ones((n, n)))
    sumyy = cv2.filter2D(src=np.multiply(Iy, Iy), ddepth=-1, kernel=np.ones((n, n)))
    sumxt = cv2.filter2D(src=np.multiply(Ix, It), ddepth=-1, kernel=np.ones((n, n)))
    sumyt = cv2.filter2D(src=np.multiply(Iy, It), ddepth=-1, kernel=np.ones((n, n)))
    sumxy = cv2.filter2D(src=np.multiply(Ix, Iy), ddepth=-1, kernel=np.ones((n, n)))

    # Compute the displacement vector (u, v)
    D = np.multiply(sumxx, sumyy) - np.multiply(sumxy, sumxy)  # Covariance matrix
    # D contains elements that appears in fractions, if they are equal to zero, we get an error.
    # Trivial solution: substitute these zeros with the machine precision of the used data.type
    D_eps = D.copy()
    D_eps[D_eps == 0] = np.finfo(D_eps.dtype).eps  # This solves division by zeros
    # All 0.000e+00 -> 2.220e-16 (eps for float64)

    u = -np.divide(np.multiply(sumyy, sumxt) - np.multiply(sumxy, sumyt), D_eps)  # horizontal displacement
    u = np.nan_to_num(u)
    v = -np.divide(np.multiply(sumxx, sumyt) - np.multiply(sumxy, sumxt), D_eps)  # vertical displacement
    v = np.nan_to_num(v)

    mask = lk_reliability(sumxx, sumyy, D, cond_th, low_th)

    show_lk_results(im1, im2, u, v, mask)

    return u, v, mask


def hornschunck(im1, im2, n_iter, lmbd):
    # Compute Ix, Iy and It
    It = gaussmooth(im2 - im1, sigma=1)  # temporal derivative
    # Ix, Iy are computed by convolution. Use the gaussderiv to compute them (one frame needed)
    Ix1, Iy1 = gaussderiv(im1, sigma=1)  # horizontal spatial derivative
    Ix2, Iy2 = gaussderiv(im2, sigma=1)  # vertical spatial derivative
    Ix = 0.5 * (Ix1 + Ix2)  # horizontal horizontal spatial derivative
    Iy = 0.5 * (Iy1 + Iy2)  # vertical horizontal spatial derivative
    # compute Ld
    Ld = np.array([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]])
    # initialize u and v as zero
    u = np.zeros(im1.shape)
    v = np.zeros(im1.shape)
    # loop on lmbd
    for k in np.arange(0, n_iter):
        # compute u_a and v_a using Ld
        u_a = cv2.filter2D(src=u, ddepth=-1, kernel=Ld)
        v_a = cv2.filter2D(src=v, ddepth=-1, kernel=Ld)
        # compute D using Ix, Iy and lmbd
        D = lmbd + Ix*Ix + Iy*Iy
        # compute P using Ix, Iy, It, u_a, v_a
        P = Ix*u_a+Iy*v_a+It
        # compute u and x using P and D
        u = u_a - Ix*(np.divide(P, D))
        v = v_a - Iy*(np.divide(P, D))

    show_hs_results(im1, im2, u, v)

    return u, v

