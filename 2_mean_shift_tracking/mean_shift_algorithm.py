import math
from ex2_utils import get_patch, generate_responses_1, generate_responses_2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm


def mean_shift(data, x_0, kernel, tol, n_max):
    """
    Mean shift algorithm: given a set of data points, it returns the mode of the
    probability distribution generated using a given kernel.
    Iterative approach to compute the mode.
    :param data: weighted dataset of points (evey element having a position and a weight)
    :param x_0: initial estimate of the mode.
    :param kernel: derivative of the kernel used to smooth the data, i.e. create the distribution
    :param tol: accepted tolerance of the result
    :param n_max: max number of iteration
    :return: estimation of the mode
    """

    # INIT:
    # First estimation of the mode
    x_k = x_0.astype('float')

    # list od mode_pos values, in the init contains only the first x_k
    mode_pos = np.array(x_k.reshape(1, 2))

    # Initial step/norm of ms_vect (initialized at something greater than tol
    ms_vect_norm = tol + 1
    # History of the steps (norms of the ms_vect)
    steps = np.array([ms_vect_norm])

    # Creation of the matrices od coordinates: x_i, y_i
    h = kernel.shape[0]  # bandwidth of the kernel
    h2 = int(np.floor(h/2))
    x_i, y_i = np.meshgrid(np.arange(-h2, h2 + 1), np.arange(-h2, h2 + 1))

    # Initialize the ms_vector that will be updated at every iteration
    ms_vect = np.empty(x_k.shape, dtype=float)

    while ms_vect_norm > tol and len(mode_pos)-1 < n_max:
        # get the patch centered at the point x_k, with diameter equal to the bandwidth
        patch, _ = get_patch(data, x_k, kernel.shape)
        # x_new is the new estimation of the shift towards the mode of the pdf
        ms_vect[0] = np.sum(patch * x_i * kernel)/(np.sum(patch * kernel)+np.finfo(np.float32).eps)
        ms_vect[1] = np.sum(patch * y_i * kernel)/(np.sum(patch * kernel)+np.finfo(np.float32).eps)

        # The error is the norm of the shift, it should progressively decrease ad reach zero
        ms_vect_norm = np.linalg.norm(ms_vect, 2)

        x_k += ms_vect  # Update the estimation of the mode

        # Storing x_k and ms_vect in the vectors containing the history of these values
        mode_pos = np.concatenate((mode_pos, x_k.reshape(1, 2)), axis=0)
        steps = np.append(steps, ms_vect_norm)

    return mode_pos.astype(np.int32), ms_vect_norm, steps[1:]


def test_mean_shift(data, x_0, kernel, tol, n_max):
    mode_pos, ms_vect, steps = mean_shift(data, x_0, kernel, tol, n_max)
    print('---- TEST ----')
    print('Estimated mode: ', mode_pos[-1, :])
    print('final error: ', ms_vect)
    print('N. of iterations: ', mode_pos.shape[0] - 1)

    # plot dataset
    plt.figure()
    ax1 = plt.axes(projection='3d')
    X, Y = np.meshgrid(np.arange(100), np.arange(100))
    ax1.plot_wireframe(X, Y, data, alpha=1, linewidth=0.5)

    # plot path of the estimation of the mode computed by mean shift
    path = np.empty(mode_pos.shape[0])
    for i in np.arange(mode_pos.shape[0]):
        x_i, y_i = mode_pos[i, :]
        path[i] = 2*data[x_i, y_i]
    ax1.plot(mode_pos[:, 0], mode_pos[:, 1], path, 'r-o', markersize=2, linewidth=0.5)

    # plot history of steps, convergence at every iteration
    plt.figure()
    plt.scatter(np.arange(mode_pos.shape[0] - 1), steps, s=5, marker='*', facecolors='none', edgecolors='r')
    plt.title('History of convergence')
    plt.xlabel("Iterations")
    plt.ylabel("Mean shift vector")
    plt.grid(True)


if __name__ == '__main__':

    # mean shift tests:
    data1 = generate_responses_1()  # generate function where to search for the maximum
    data2 = generate_responses_2()
    # test_mean_shift(data1, x_0=np.array([30, 50]), kernel= np.ones((9, 9)), tol=0.1, n_max=500)
    # test_mean_shift(data2, x_0=np.array([60, 70]), kernel=np.ones((9, 9)), tol=0.1, n_max=500)
    test_mean_shift(data1, x_0=np.array([40, 40]), kernel=np.ones((5, 5)), tol=.01, n_max=500)
    test_mean_shift(data2, x_0=np.array([40, 40]), kernel=np.ones((31, 31)), tol=.01, n_max=500)

    plt.show()