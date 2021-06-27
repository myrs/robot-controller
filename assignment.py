import numpy as np
import numpy.linalg.linalg as LA
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.transforms import Affine2D
import matplotlib.patches as patches
from matplotlib.text import Annotation
import scipy.stats as stats

from utils import angle_between

twoPi = np.pi * 2


class MultipleNRV(object):
  """Multiple independent normal random variables."""

  def __init__(self, size, loc=0., scale=1.):
    self.size = size
    self.mean, self.std = loc, scale
    self.twoVariance = 2 * self.std ** 2

  def pdf(self, xs):
    """Returns the probability density function value for a particular
    vector."""
    twoVar = self.twoVariance

    if twoVar == 0:
      return 1 if xs == self.mean else 0
    else:
      delta2 = (xs - self.mean) ** 2

      return np.product(np.exp(-delta2 / twoVar) / np.sqrt(twoVar * np.pi))

  def sample(self):
    """Returns a vector sampled from the PDF."""
    loc, scale, n = self.mean, self.std, self.size
    return loc if scale == 0 else np.random.normal(loc, scale, size=self.size)


class World(object):

  def __init__(self, sensor_angles=(0.0,), luminance=1.0,
               light_coords=(10.0, 0.0, -0.1), v_max=1.0, agent_radius=0.5,
               sensor_noise=0.01, motor_noise=0.5, dt=0.1, seed=None, within=1.5,
               random_orientation=True, orientation=0, pert_motor_gain=[1, 1],
               pert_angle_change=0, pert_gradual_angle_change=0,
               pert_light_position_constant=(0, 0, 0), pert_light_position=0,
               pert_luminance=0):

    self.sensors = np.array(sensor_angles)
    self.light_pos = np.array(light_coords)
    self.v_max = v_max
    self.agent_radius = agent_radius

    self.luminance = luminance
    self.luminance_default = luminance
    self.pert_luminance = pert_luminance

    self.dt = dt
    self.within = within
    self.random_orientation = random_orientation
    self.orientation = orientation
    self.pert_motor_gain = pert_motor_gain
    self.pert_angle_change = pert_angle_change
    self.pert_gradual_angle_change = pert_gradual_angle_change
    self.angle_change_cum = 0
    self.pert_light_position_constant = pert_light_position_constant
    self.pert_light_position = pert_light_position
    self.reached_light_at_v = np.inf

    self.light_pos_reached = np.array([np.inf, np.inf, -0.1])

    if seed is not None:
      np.random.seed(seed)

    # set up noise random variables
    sensor_sigma = sensor_noise * np.sqrt(dt)
    motor_sigma = motor_noise * np.sqrt(dt)
    self.sensor_rv = MultipleNRV(size=len(sensor_angles), scale=sensor_sigma)
    self.motor_rv = MultipleNRV(size=2, scale=motor_sigma)

  def sensor_pos(self, state):
    """Returns an array corresponding to a list of (x, y, 0) sensor
    positions in world coordinates."""

    # # PERTURBATION gradual sensor position change
    self.sensors += self.pert_gradual_angle_change

    sensors, r = self.sensors, self.agent_radius
    x, y, theta = state
    n = len(sensors)

    result = np.zeros((n, 3))

    # copy robot x, y into sensors
    result[:, 0:2] = state[0:2]

    # PERTURBATION sudden angle change
    if self.pert_angle_change:
      angle_change_sudden = np.random.random() * self.pert_angle_change
    else:
      angle_change_sudden = 0
      angles = theta + sensors + angle_change_sudden

    # angles = theta + sensors
    result[:, 0] = r * np.cos(angles) + x
    result[:, 1] = r * np.sin(angles) + y

    return result

  def sensor_input(self, state):
    """Returns an array of raw sensor input values for a particular
    agent state (position and orientation). These are calculated
    according to an inverse square distance law, and the agent's body
    can occlude a sensor reducing its input to zero.
    """

    # PERTURBATION gradual light position change
    self.light_pos += self.pert_light_position_constant

    if self.pert_light_position:
      perturbation = (0, self.pert_light_position * self.dt, 0)
      self.light_pos += perturbation

    # various relevant parameters
    r, K = self.agent_radius, self.luminance
    # light position
    l_pos = self.light_pos

    # unpack 3D position and heading from (x, y, theta) state
    pos, theta = np.array(tuple(state[0:2]) + (0,)), state[-1]

    # positions in world coordinates of each sensor
    s_pos = self.sensor_pos(state)
    # array of distances of sensors from light source
    d_s = LA.norm(l_pos - s_pos, axis=1)

    # distance of light from robot's centre
    d_0 = LA.norm(l_pos - pos)

    # array of zeros or ones for each sensor according to whether the
    # agent's body lies between the sensor and the light source
    not_occluded = (d_0**2 >= r**2 >= (d_s**2 - d_0**2))

    # light reaching each sensor
    return not_occluded * K / d_s ** 2

  def sensor_transform(self, activation):
    """Returns a vector of sensor readings for a particular sensor input
    value (activation) vector. Noise is usually applied to the activation
    before applying the transform."""
    # rescale to (0, 1) interval, assuming activation is positive
    # return activation / (1 + activation)

    K, l_pos = self.luminance, self.light_pos
    # minimum distance is z coordinate of the light position
    d_min = l_pos[-1]

    # rescale activation to range between 0 and a_max
    # with midpoint around
    a_max = K / (d_min ** 2)
    a = a_max / (1 + np.exp(5 * (K / 4 - activation)))

    # return 1 / (1 + np.exp(-activation))
    return activation
    # return np.sqrt(K / a)

  def sensor_inverse_transform(self, reading):
    """Returns the vector of sensor input values (activations) that would be
    needed to produce the specified sensor reading. """
    return reading / (1 - reading)

  def sense(self, state):
    """Returns a vector of sensor reading values for a
    particular agent state (position and orientation).
    Noise is added to the raw luminance at the sensor's location
    and the result is rescaled to the (0, 1) interval.
    """
    activation = self.sensor_input(state) + self.sensor_rv.sample()

    # and rescale to (0, 1) interval
    return self.sensor_transform(activation)

  def p_sensation(self, state, sensation):
    """Returns a probability density value for the likelihood of a
    particular sensor reading vector given a particular agent state."""
    # invert rescaling operation to find the original activations
    sensor_activation = self.sensor_inverse_transform(sensation)
    # determine the actual luminance at the sensors
    sensor_input = self.sensor_input(state)

    # interrogate the RV object to get the PDF value
    return self.sensor_rv.pdf(sensor_input - sensor_activation)

  def act(self, state, action):
    """Applies a motor activation vector to an agent state, and simulates
    the consequences using Euler integration over a dt interval."""
    # noisily map the action values to a (-1, +1) interval
    motor_out = self.v_max * np.tanh(action) + self.motor_rv.sample()

    # # PERTURBATION constant motor gain change
    motor_out[0] = motor_out[0] * self.pert_motor_gain[0]
    motor_out[1] = motor_out[1] * self.pert_motor_gain[1]

    # calculate the linear speed and angular speed
    v = motor_out.mean()
    omega = (motor_out[1] - motor_out[0]) / (2.0 * self.agent_radius)

    # calculate time derivative of state
    theta = state[-1]
    deriv = [v * np.cos(theta), v * np.sin(theta), omega]

    # calculate the linear speed and angular speed v = motor_out.mean()
    omega = (motor_out[1] - motor_out[0]) / (2.0 * self.agent_radius)

    # calculate time derivative of state theta = state[-1]
    deriv = [v * np.cos(theta), v * np.sin(theta), omega]

    # perform Euler integration
    return self.dt * np.array(deriv) + state

  def update_perturbations(self):  # perturb
    # PERTURBATION luminance if self.pert_luminance:
    if self.pert_luminance:
      self.luminance = self.luminance_default

      if np.random.random() > 0.5:
        change = -self.pert_luminance + self.pert_luminance * np.random.random()
      else:
        change = 0

      self.luminance += change

  def simulate(self, controller, interval=500.0):
    """Simulates the agent-environment system for the specified interval
    (in simulated time units) starting from a random state. Returns
    a (poses, sensations, actions, states) tuple where poses is a time array of agent poses (position and orientation), sensations is a time array of sensory readings, actions is a time array of motor activations, and states is a list of arbitrary internal controller state objects.

    Must be called with a controller function of the form controller(sensation, state, dt) that returns a (action, state) tuple outputting motor activations and updated internal state in
    response to sensor readings. """

    poses = [self.random_state()]
    states = [False]
    sensations = []
    actions = []
    reached = False

    for i in range(int(interval / self.dt)):

      if not reached:
        ds = LA.norm(self.light_pos[0:2] - poses[-1][0:2])
        if ds < self.within:
          # if reached the light store the position of the light
          # to use in calculations of the second fitness
          # and show later in animation
          # as the light can be moved by the perturbation
          # this adjustment needs to be done to the simulation reached = True
          self.light_pos_reached = np.array(self.light_pos)
          self.reached_light_at_v = i

          action, state = controller.output(sensations[-1], states[-1])
          actions.append(action)
          states.append(state)
          poses.append(self.act(poses[-1], actions[-1]))

    if not reached:
      self.light_pos_reached = self.light_pos

    return np.array(poses), np.array(sensations), np.array(actions), states

  def random_state(self):
    """Returns a random initial state."""
    result = np.zeros(3)
    if self.random_orientation:
      result[-1] = np.random.rand() * twoPi
    else:
      result[-1] = self.orientation

    return result

  def after_to_index(self, after):
    return int(np.floor(after / self.dt))

  def fitness_path_covered(self, poses, target, after=0):
    # calculate min distance between the desired target and poses
    min_distance = self.min_distance(poses, target, after=after)

    after_index = self.after_to_index(after)
    initial_distance = LA.norm(target - poses[after_index, 0:2])
    # print('initial_distance')
    # print(initial_distance)

    # reward for minimizing the distance positive fitness # max is 1, min is 0
    # (self.within is subtracted from distance so need to so it here too)
    path_covered = 1 - min_distance / (initial_distance - self.within)

    return path_covered

  def fitness_speed(self, poses, target, after=0):
    reached_at = self.first_reached(poses, target, after=after)

    if reached_at != np.inf:
      # scaling constant (10) so around 1 will be maximum
      return 10 / reached_at

    return 0

  def fitness(self, poses, target, after=0):  # Two components
    # first check target is reachable
    # 1. Minimum distance score (max = 1)
    path_not_covered_punish = self.fitness_path_covered(poses, target, after) - 1

    # 2. Reached the desired target score
    # second when desired target was reached
    speed_reward = self.fitness_speed(poses, target, after)
    reward = path_not_covered_punish + speed_reward
    return reward

  def task1fitness(self, poses):
    """Returns the fitness of the trajectory described by poses 
    on assignment task 1 (reaching the light source)."""
    light_time = self.reached_light_at(poses)

    after_light = light_time - 1 if light_time != np.inf else 0
    return self.fitness(poses, self.light_pos_reached[0:2], after=after_light)

  def task2fitness(self, poses):
    """Returns the fitness of the trajectory described by poses on assignment 
    task 1 (reaching the light source and returning to base)."""
    light_time = self.reached_light_at(poses)

    if light_time == np.inf:
      return 0

    # this was the last step
    if self.after_to_index(light_time) == len(poses) - 1:
      return 0

    return self.fitness(poses, np.array([0, 0]), after=light_time)

  def task1fitness_detailed(self, poses):
    """ return tuple: reached or not, how fast, how close it got"""
    light_time = self.reached_light_at(poses)

    after_light = light_time - 1 if light_time != np.inf else 0

    # how close did I get
    path_covered = self.fitness_path_covered(
        poses, self.light_pos_reached[0:2], after=after_light)

    # reached or not
    reached = light_time != np.inf

    return reached, path_covered, light_time

  def task2fitness_detailed(self, poses):
    """ return tuple: reached or not, how fast, how close it got"""
    home = np.array([0, 0])
    light_time = self.reached_light_at(poses)

    if light_time == np.inf:
      return False, 0, np.inf

    # how close did I get
    path_covered = self.fitness_path_covered(poses, home, after=light_time)

    # how fast it got
    home_time = self.first_reached(poses, home, after=light_time)

    # reached or not
    reached = home_time != np.inf

    return reached, path_covered, home_time

  def min_distance(self, poses, point, after=0):
    after_index = self.after_to_index(after)
    ds = LA.norm(point - poses[after_index:, 0:2], axis=1)

    min_distance = ds.min()
    # minimal distance is 0 if it's less, that within interval
    min_distance = min_distance - self.within if min_distance > self.within else 0

    return min_distance

  def first_reached(self, poses, xy, after=0):
    after_index = self.after_to_index(after)
    ds = LA.norm(xy - poses[after_index:, 0:2], axis=1)
    indices = np.nonzero(ds < self.within)[0]

    if len(indices) == 0:
      return np.inf

    return (indices[0] + after_index) * self.dt

  def reached_light_at(self, poses):
    """ return second the light was reached """
    # return self.first_reached(poses, self.light_pos[0:2])
    return self.reached_light_at_v * self.dt
    # return self.first_reached(poses, self.light_pos_reached[0:2])

  def animate(self, poses, sensations, speedup=5):
    # r, l_pos = self.agent_radius, self.light_pos
    r, l_pos = self.agent_radius, self.light_pos_reached

    x, y, theta = poses[0]

    # use an Ellipse to visually represent the agent's body
    body = patches.Ellipse(xy=(0, 0), width=2 * r, height=2 * r, fc='w', ec='k')

    # use a black dot to visually represent each sensor
    sensors = [patches.Ellipse(xy=(r * np.cos(theta), r * np.sin(theta)),
                               width=0.2, height=0.2, fc='b') for theta in self.sensors]

    # use small rectangles to visually represent the motors
    motors = [patches.Rectangle(
        (-0.5 * r, y), width=r, height=0.2 * r, color="black")
        for y in (-1.1 * r, 0.9 * r)]

    # use a line to indicate the agent's orientation
    line = Line2D((x, x + r * np.cos(theta)), (y, y + r * np.sin(theta)))
    line = Line2D((0, r), (0, 0))
    # draw a line showing the agent's "trail" trail = Line2D([], [], color='r')
    # display a clock

    clock = Annotation('', (0.8, 0.9), xycoords='axes fraction')

    # use a yellow circle to visually represent the light
    light_r = patches.Ellipse(xy=l_pos[0:2], width=1, height=1, fc='y', ec='none')
    light = patches.Ellipse(xy=l_pos[0:2], width=0.25, height=0.25, fc='b')

    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [10, 1]})

    ax1.axis("equal")
    ax1.set_xlim([-15, 15])
    ax1.set_ylim([-15, 15])
    ax1.set_title("Click On Main Display To Pause / Unpause")
    ax2.set_title("Click On Sensor Graph To Change Time")
    tracker = ax2.axvline(0, 0, 1)
    paused = [False]
    last_index = [-1]
    t_index = [0]

    if sensations is not None:
      times = np.arange(0, self.dt * len(sensations), self.dt)  # plot the sensor values
      ax2.plot(times, sensations, 'k')
      # plot the ideal (noiseless) sensor values
      ideal = np.array([self.sensor_transform(self.sensor_input(pose))
                        for pose in poses[:-1]])
      ax2.plot(times, ideal, 'r')

    def draw(index):
      if not paused[0]:
        t_index[0] = t_index[0] + (index - last_index[0])
        t_index[0] = t_index[0] % len(poses)

      last_index[0] = index

      x, y, theta = poses[t_index[0]]
      tr = Affine2D().rotate(theta).translate(x, y) + ax1.transData
      agent_patches = (body, line) + tuple(sensors) + tuple(motors)
      for patch in agent_patches:
        patch.set_transform(tr)

      trail.set_data(poses[:t_index[0], 0], poses[:t_index[0], 1])
      time = t_index[0] * self.dt
      tracker.set_xdata([time, time])

      # clock.set_text("Time: %.02f" % time)
      return (trail, light_r, light, clock, tracker) + agent_patches

    def init():
      result = draw(0)

      for artist in result:
        if artist is not tracker:
          ax1.add_artist(artist)

      return result

    def onclick(event):
      if event.button == 1:
        # pause if the user clicks on the main figure
        if event.inaxes is ax1:
          paused[0] = not paused[0]
          print('paused')

        # edit time directly if the user clicks on the graph over time
        elif event.inaxes is ax2:
          t_index[0] = (int)(event.xdata / self.dt)

    def anim(index):
      return draw(index)

    ani = FuncAnimation(fig, anim, init_func=init,
                        frames=None, interval=1000 * self.dt / speedup,
                        blit=True, save_count=len(poses))

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.ion()
    plt.show()

    return ani
