import numpy as np
from my_utils import kalman_step
import sympy as sp
from sympy.interactive.printing import init_printing
from matplotlib import pyplot as plt


# function that implements the kalman filter:
def kalman_filter(x, y, A, C, Q_i, R_i):
    # Takes as input:
    # - x and y: coordinates of the observations (measures)
    # - A: system matrix (Fi), depends on the chosen motion model
    # - C: observation matrix (H), depends on the chosen motion model
    # - Q_i: initial value of the variance for the dynamic model
    # - R_i: initial value of the variance for the observation model
    # Returns as output the history of states

    # SETUP/INIT ----------------------------

    # creates the variables for the history of the state
    sx = np.zeros((x.size, 1), dtype=np.float32).flatten()
    sy = np.zeros((y.size, 1), dtype=np.float32).flatten()

    # initial state is the first measure/observation (at the beginning the state is the measure itself)
    sx[0] = x[0]
    sy[0] = y[0]

    # temp variable to store the current state
    state = np.zeros((A.shape[0], 1), dtype=np.float32).flatten()

    # LOOP/STATE ESTIMATION -----------------

    # first state
    state[0] = x[0]
    state[1] = y[0]

    # initialization of the covariance fo the prior pdf, initially is an identity matrix
    prior_covariance = np.eye(A.shape[0], dtype=np.float32)

    # For every observation
    for j in range(1, x.size):
        # compute the new state and prior covariance using the kalman step function
        # state, covariance = kalman_step(A, C, Q, R, y, x, V)
        observations = np.reshape(np.array([x[j], y[j]]), (-1, 1))
        state = np.reshape(state, (-1, 1))
        state, prior_covariance, _, _ = kalman_step(A, C, Q_i, R_i, observations, state, prior_covariance)
        # store the computed state in the history of states
        sx[j] = state[0]
        sy[j] = state[1]

    return sx, sy


# trajectory generations
def generate_spiral_trajectory(noise_var=0):  # from the instructions
    N = 50
    v = np.linspace(5 * np.pi, 0, N)
    x = np.cos(v) * v
    y = np.sin(v) * v
    # Add noise
    if noise_var != 0:
        noise_x = np.random.normal(0, noise_var, len(x))
        noise_y = np.random.normal(0, noise_var, len(y))
        x = x + noise_x
        y = y + noise_y
    return x, y


def generate_exponential_trajectory(x_noise_var=0, y_noise_var=0):  # from the instructions
    N = 50
    x = np.linspace(0, 10, N)
    y = np.exp(x)
    # Add noise
    if x_noise_var != 0 or y_noise_var != 0:
        noise_x = np.random.normal(0, x_noise_var, len(x))
        noise_y = np.random.normal(0, y_noise_var, len(y))
        x = x + noise_x
        y = y + noise_y
    return x, y


# Motion model parameters generations
def generate_parameters(model, q, r):
    # variables initialization
    # - T (time step)
    # - q (dynamic system variance)
    # - F (dynamic system matrix)
    # - L (dynamic system matrix)
    # - H (observation model matrix)
    T_sym, q_sym = sp.symbols('T q')
    F_sym = 0
    L_sym = 0
    H = 0

    # Filling F, L, H depending on the motion model
    if model == 'rw':
        F_sym = sp.Matrix([[0, 0],
                           [0, 0]])
        L_sym = sp.Matrix([[1, 0],
                           [0, 1]])
        H = np.array([[1, 0],
                      [0, 1]])
    elif model == 'ncv':
        F_sym = sp.Matrix([[0, 0, 1, 0],
                           [0, 0, 0, 1],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0], ])
        L_sym = sp.Matrix([[0, 0],
                           [0, 0],
                           [1, 0],
                           [0, 1]])
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])
    elif model == 'nca':
        F_sym = sp.Matrix([[0, 0, 1, 0, T_sym, 0],
                           [0, 0, 0, 1, 0, T_sym],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0]])
        L_sym = sp.Matrix([[0, 0],
                           [0, 0],
                           [0, 0],
                           [0, 0],
                           [1, 0],
                           [0, 1]])
        H = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0]])
    else:
        print('not found')
        exit(-1)

    # Computation of Fi and Q
    # - Fi (motion model matrix)
    # - Q (motion model covariance matrix)
    Fi_sym = sp.exp(F_sym * T_sym)
    Q_sym = sp.integrate((Fi_sym * L_sym) * q_sym * (Fi_sym * L_sym).T, (T_sym, 0, T_sym))

    # Conversion from symbolic to numpy array
    get_Fi = sp.lambdify(T_sym, Fi_sym, modules="numpy")
    get_Q = sp.lambdify((T_sym, q_sym), Q_sym, modules="numpy")

    # Get Fi, Q and R with T(time step)=1
    # - Fi (motion model matrix)
    # - Q (motion model covariance matrix)
    # - R (observation model covariance matrix)
    Fi = get_Fi(1)
    Q = get_Q(1, q)
    R = r * np.eye(2, dtype=np.float32)
    # R is a diagonal matrix since the variance is equal fot the 2 components of the observations
    return Fi, H, Q, R


# example
def single_test(measures, model, q, r, axes):
    Fi_ncv, H_ncv, Q_ncv, R_ncv = generate_parameters(model, q, r)
    sx_1, sx_2 = kalman_filter(measures[0], measures[1], Fi_ncv, H_ncv, Q_ncv, R_ncv)
    # plotting trajectories
    axes.plot(measures[0], measures[1], "-ro", label="Measurements", ms=3)
    axes.plot(sx_1, sx_2, "-bo", label="Kalman filter correction", ms=3)
    axes.set_title(f"{model}) q = {q}, r = {r}")


def multiple_tests(measures):
    # parameters
    q_values = (50., 10., 1., 1., 1., 1., 1)
    r_values = (1., 1., 1., 10., 50., 100., 200.)
    models = ('rw', 'ncv', 'nca')

    fig, axes = plt.subplots(3, len(q_values), figsize=(16, 10))
    for i in np.arange(len(q_values)):
        for j in np.arange(len(models)):
            single_test(measures, models[j], q_values[i], r_values[i], axes[j, i])
    plt.show()


if __name__ == '__main__':
    measures = generate_spiral_trajectory(noise_var=0)
    multiple_tests(measures)
    measures = generate_spiral_trajectory(noise_var=0.5)
    multiple_tests(measures)
    measures = generate_exponential_trajectory(x_noise_var=0, y_noise_var=5e2)
    multiple_tests(measures)