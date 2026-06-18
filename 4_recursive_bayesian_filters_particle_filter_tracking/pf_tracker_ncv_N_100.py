from kalman_filter_motion_models import generate_parameters
from my_utils import create_epanechnik_kernel, get_patch, extract_histogram
from my_utils import Tracker
# from utils.tracker import Tracker
import numpy as np
from matplotlib import pyplot as plt


# PARTICLE FILTER TRACKER ----------------------------------------------------------------------------------------------
# PF TRACKER inherits from the Tracker class
class PFTracker_ncv_N_100(Tracker):

    @staticmethod
    def name():
        return "PFTracker_ncv_N_100"

    # METHODS USED IN THE TRACKING FUNCTION: ---------------------------------------------------------------------------

    # The visual model of a given region is retrieved:
    # - during initialization, to get the target histogram
    # - during a tracking iteration, to update the weights of the N particles
    # - at the end of a tracking iteration, to update the target histogram
    def get_region_visual_model(self, image, region_center):
        # Crop the image in that region to get the patch
        search_region, _ = get_patch(image, region_center, self.kernel.shape)
        # Compute the weighted histogram
        hist = extract_histogram(search_region, n_bins=self.parameters.n_bins, weights=self.kernel)
        hist = np.divide(hist, np.sum(hist))
        return hist

    # The particles are generated during initialization, as samples of a gaussian distribution centered at the target
    # center position, with standard deviation Q. The weights are all set to 1
    def generate_particles(self, model):
        # initially, the particles are all in the center, we apply a gaussian noise with covariance Q to move them from
        # the center
        particle_state = np.array([[0],
                                   [0]])
        if model == 'rw':
            particle_state = np.array([[self.target_position[0]],   # x
                                       [self.target_position[1]]])  # x
        elif model == 'ncv':
            particle_state = np.array([[self.target_position[0]],   # x
                                       [self.target_position[1]],   # y
                                       [0],                         # x'
                                       [0]])                        # y'
        elif model == 'nca':
            particle_state = np.array([[self.target_position[0]],   # x
                                       [self.target_position[1]],   # y
                                       [0],                         # x'
                                       [0],                         # y'
                                       [0],                         # x"
                                       [0]])                        # y"
        else:
            print('not found')
            exit(-1)
        # Copy the first particle N times to get N equal particles
        particles_states = np.tile(particle_state, (1, self.parameters.n_particles))
        # Generate a vector of gaussian noise N(0,Q)
        noise_term = np.random.multivariate_normal(np.zeros(self.Q.shape[0]), self.Q, self.parameters.n_particles).T
        # Add the noise to the particle initial position
        particles_states[0, :] = particles_states[0, :] + noise_term[0, :]
        particles_states[1, :] = particles_states[1, :] + noise_term[1, :]
        # Initialize the particles states
        self.particles_states = particles_states
        # Initialize the particles weights
        self.particles_weights = np.ones(self.parameters.n_particles)

    def initialize(self, image, region):
        # image is the initialization frame
        # region is a list of 4 coordinates identifying the target region

        # This line is needed to set the parameters at the default values
        # self.parameters = PFParams_ncv_N_100()

        # Get the central position of the region (center of the target)
        self.target_position = (round(region[0] + region[2] / 2), round(region[1] + region[3] / 2))
        # Get the region size (make sure the region is odd sized)
        self.region_size = (int((np.ceil(region[2] // 2) * 2 + 1)),
                            int((np.ceil(region[3] // 2) * 2 + 1)))

        # GET THE VISUAL MODEL OF THE TARGET ---------------------------------------------------------------------------
        # This is used to evaluate the observation model
        # The observation model is the likelihood function p(y_k|x_k) computed from the Hellinger distance between the
        # target histogram and a particle histogram
        # ----------------------------------------------------
        # Compute the Epanechnikov kernel with the region size
        self.kernel = create_epanechnik_kernel(self.region_size[1], self.region_size[0], self.parameters.kernel_param)
        self.target_hist = self.get_region_visual_model(image, self.target_position)

        # DEFINITION OF THE PARTICLES GIVEN THE MOTION MODEL -----------------------------------------------------------
        # If paramters.q=1, the sd of the gaussian distribution that represents the noise applied to the dynamic model
        # will be half of the size of the target region
        self.q = self.parameters.q*(min(self.region_size)//2)
        # Retrieve the motion model matrices
        self.Fi, _, self.Q, _ = generate_parameters(self.parameters.motion_model, self.q, 1)
        # Define and generate the particles given the motion model
        self.generate_particles(self.parameters.motion_model)

    # METHODS USED IN THE TRACKING FUNCTION: ---------------------------------------------------------------------------
    def particles_resampling(self):  # (1)
        # Before sampling, the particles have different weights and form a wighted Dirac's deltas mixture model,
        # which is our posterior p(x_(k-1)|y_(1:k-1)).
        # At the beginning, the weights will be all equal to one, in the successive iterations the weights will be the
        # We extract N new samples form this distribution. The new samples will have equal wights.
        states = self.particles_states
        weights = self.particles_weights
        weights_norm = weights / np.sum(weights)  # normalize weights
        weights_cumsumed = np.cumsum(weights_norm)  # cumulative distribution
        rand_samples = np.random.rand(self.parameters.n_particles, 1)
        sampled_idxs = np.digitize(rand_samples, weights_cumsumed)  # randomly select N indices
        particles_new_states = states[:, sampled_idxs.flatten()]  # select the corresponding samples
        self.particles_states = particles_new_states

    def motion_model_application(self):  # (2)
        # With the new N samples (particles) sampled from the previous posterior, we move every particles with the
        # given motion model
        # x_k^(i) = Fi * x_k^(i-1) + w_k^(i)
        # Fi is the deterministic shift (linear motion)
        # w_k^(i) is the gaussian noise N(0,Q)
        # -------------------------------------------------
        # get particles states
        states = self.particles_states
        # compute the noise term by sampling a gaussian noise N(0,Q)
        noise_term = np.random.multivariate_normal(np.zeros(self.Q.shape[0]), self.Q, self.parameters.n_particles).T
        # apply the motion model
        self.particles_states = np.matmul(self.Fi, states) + noise_term

    def particle_weights_update(self, image):  # (3)
        # After the application of the motion model, we have a distribution of particles with the same weights, in new
        # different positions. This prediction is corrected using the observation model p(y_k|x_k) for every particle,
        # that acts as the weight of the particle.
        # The observation model p(y_k|x_k) is the likelihood, it is computed by the Hellinger distance between the
        # visual model of the target and the visual model obtained by the observed particle

        sigma = 0.1  # standard deviation of the gaussian pdf produced from the Hellinger distance

        # updated weights initialization
        # new_weights = np.zeros(len(self.particles_weights))

        # weights update
        for i in np.arange(self.particles_states.shape[1]):
            # if the particle position is out of the image
            if self.particles_states[0, i] < 0 or self.particles_states[0, i] > image.shape[1] or \
                    self.particles_states[1, i] < 0 or self.particles_states[1, i] > image.shape[0]:
                # the weight is set to zero
                self.particles_weights[i] = 0
            # else, if the particle is in the image
            else:
                # Get the histogram of the region centered at the particle position
                particle_hist = self.get_region_visual_model(image, region_center=self.particles_states[:2, i])
                # particle_hist is normalized: is a pdf, we call it p(x)
                # self.target_hist is normalized: is a pdf, we call it q(x)
                # Bhattacharyya measure: rho(x) = sum_u((p_u*q_u)^(1/2))
                rho = np.sum(np.sqrt(particle_hist*self.target_hist))
                # Hellinger distance: h_dist(x) = 2-2*rho(x)
                h_dist = 2-2*rho
                # Alternative way of computing the Hellinger distance
                # h_dist = (1 / np.sqrt(2)) * np.linalg.norm(np.sqrt(particle_hist) - np.sqrt(self.target_hist))
                # Sets as weight the value of p(y_k|x_k), computed converting the distance in a value from a gaussian
                # distribution
                self.particles_weights[i] = np.exp(-0.5*(np.square(h_dist)/np.square(sigma)))

    def get_new_target_position(self):  # (4)
        # In the case the sum of the weights is zero, we get true division error and the position results (NaN, Nan)
        if np.sum(self.particles_weights) != 0:
            weights = self.particles_weights / np.sum(self.particles_weights)
            x_new = np.sum(np.multiply(weights, self.particles_states[0, :]))
            y_new = np.sum(np.multiply(weights, self.particles_states[1, :]))
            # Update estimated target position
            self.target_position = (x_new, y_new)
            # print('new position:', self.target_position)
        # else:
            # nothing, don't update the target position

    def update_target_visual_model(self, image):  # (5)
        # Having the new target position, we can compute the visual model for the target in the current frame
        new_target_hist = self.get_region_visual_model(image, region_center=self.target_position)
        # We update the target visual model as a weighted average between the previous model and the new visual model
        # This is a constant updating through time, it is obvious that this leads to problem with occlusion
        # It would be possible to use not constant updating, here we just set the update_speed at a low value in order
        # to avoid fast learning, that would lead to problems with occlusions.
        self.target_hist = (1-self.parameters.alpha)*self.target_hist + self.parameters.alpha*new_target_hist

    def track(self, image):
        # five main steps:
        #                  (1) resampling,
        #                  (2) application of the motion model on the particles,
        #                  (3) particle wights update,
        #                  (4) computing new position,
        #                  (5) update the visual model of the target for robustness

        # Visualization of the particles
        # plt.figure()
        # plt.imshow(image)
        # plt.plot(self.particles_states[0, :], self.particles_states[1, :], "ro", ms=1)
        # plt.show()

        self.particles_resampling()              # (1)
        self.motion_model_application()          # (2)
        self.particle_weights_update(image)      # (3)
        self.get_new_target_position()           # (4)
        self.update_target_visual_model(image)   # (5)

        # Returns the coordinates of the new estimated region
        return [self.target_position[0] - self.region_size[0] / 2.0,
                self.target_position[1] - self.region_size[1] / 2.0,
                self.region_size[0],
                self.region_size[1]]


# class of the parameters of the tracker, here we generate all the parameters with a custom or a default value.
class PFParams_ncv_N_100():
    def __init__(self, motion_model='ncv', q=1, kernel_param=1, n_bins=8, alpha=0.05, n_particles=100):
        # Dynamic model parameters
        self.motion_model = motion_model  # a string for choosing the motion model
        # Fi (system matrix) and Q (noise covariance matrix) are retrieved once the motion model is chosen
        self.q = q  # variance of the noise applied to the motion model
        # Observation model parameters
        self.kernel_param = kernel_param  # sigma parameter fo the epanechnikov kernel
        self.n_bins = n_bins  # number of bins of the color histograms
        self.alpha = alpha  # update parameter of the visual model
        # Tracker parameters
        self.n_particles = n_particles  # number of particles used in the tracker


if __name__ == '__main__':
    pass
