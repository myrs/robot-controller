import numpy as np
from random import randint

from multiprocessing import Process, Queue
from utils import queue_to_array, get_generation_genotypes, get_individual

from assignment import World
from ctrnn import CTRNN


class RobotEmbryology:
  def __init__(self,
               controller_1_file=None,
               controller_1_genotype=None,
               dt=0.1,
               hidden_nodes=8,
               interval=100,
               min_stability=None,
               trials=10,
               use_genotype_file=None,
               ignore_inputs=False,
               world_params={}):

    self.dt = dt
    self.interval = interval
    self.min_stability = min_stability
    self.trials = trials
    self.use_genotype_file = use_genotype_file
    self.total_nodes = hidden_nodes
    self.world_params = world_params
    self.ignore_inputs = ignore_inputs

    # second task if controller 1 file or genotype were provided
    self.task = 2 if controller_1_file or controller_1_genotype else 1

    self.controller_inputs = 1 if self.task == 1 else 4

    if self.task == 2:

      if controller_1_genotype is None:
        controller_1_genotype, _ = get_individual(controller_1_file)

      controller_1_genotype = np.array(controller_1_genotype)
      controller_1_params = self.unpack_genotype(controller_1_genotype, nodes=8)
      self.controller_1 = CTRNN(*controller_1_params,
                                total_time=self.dt,
                                time_delta=self.dt)

    if self.use_genotype_file:
      self.loaded_genotypes = get_generation_genotypes(self.use_genotype_file)

  def take_genes(self, genotype, start, number):
    new_start = start + number
    return genotype[start: start + number], new_start

  def unpack_genotype(self, genotype, nodes=None, inputs=1):
    # start with gene position 0
    pos = 0

    weights, pos = self.take_genes(genotype, pos, nodes ** 2)
    tau, pos = self.take_genes(genotype, pos, nodes)
    theta, pos = self.take_genes(genotype, pos, nodes)

    inputs_gain = np.array([1.0, 1.0, 1.0, 1.0])
    for i in range(inputs):
      input_gain, pos = self.take_genes(genotype, pos, 1)
      # scale from [0, 1] to [0.1, 10]
      input_gain = 0.1 + input_gain[0] * 9.9
      inputs_gain[i] = input_gain

    motor_gain, pos = self.take_genes(genotype, pos, 1)

    w = weights.reshape(nodes, nodes)
    # scale the values of the weights from [0, 1] to [-10, 10]
    w = -10 + w * 20

    # scale the values of the time constant from [0, 1] to [0.1, 5]
    tau = 0.1 + tau.reshape(nodes, 1) * 4.9

    # scale the values of the biases from [0, 1] to [-5, 5]
    theta = -5 + theta.reshape(nodes, 1) * 10

    # scale from [0, 1] to [0.1, 10]
    motor_gain = 0.1 + motor_gain * 9.9

    return w, tau, theta, inputs_gain, motor_gain

  def run_simulation(self, fitnesses_q, w, tau, theta, inputs_gain,
                     motor_gain, orientation, motor_noise=0.5, sensor_noise=0.01):
    # force numpy to reseed inside the process
    # without there the same seed with be always used np.random.seed()

    world = World(dt=self.dt, random_orientation=False, orientation=orientation,
                  motor_noise=motor_noise, sensor_noise=sensor_noise,
                  **self.world_params)

    self.contoroller = CTRNN(w, tau, theta, inputs_gain=inputs_gain,
                             motor_gain=motor_gain, total_time=self.dt,
                             time_delta=self.dt)

    poses, sensations, actions, states = world.simulate(self, interval=self.interval)

    if self.task == 1:
      fitness = world.task1fitness(poses)
    else:
      world.task2fitness(poses)
      fitnesses_q.put(fitness)

  def output(self, sensor, reached_light):
    """ Produces output"""

    # evolution for task 1
    if self.task == 1:
      motors, _ = self.contoroller.output(input_1=sensor)
      return motors, None

    # evolution for task 2

    # mode 1 - haven't reached the light yet # print(sensor)
    # update reaching the light if haven't reached the light
    if not reached_light:
      reached_light = sensor[0] > 1

    if not reached_light:
      motors, _ = self.controller_1.output(input_1=sensor)
      motor_1, motor_2 = motors

      # if evolving standalone controller
      if not self.ignore_inputs:
        self.contoroller.output(input_2=sensor, input_3=motor_1, input_4=motor_2)

      return motors, reached_light

    # mode 2 - have reached the light
    else:
      motors, _ = self.contoroller.output(input_1=sensor)
      return motors, reached_light

  # mean is 1 - robot reached the target
  def get_stability_coefficient(self, stability, force=1):
    if stability > self.min_stability:
      print('Minimal stability!')
      multiplier = force / (1 - self.min_stability)
      stability_coeff = np.e ** ((stability - self.min_stability) * multiplier)
      return stability_coeff

    else:
      return 1

  def calculate_fitness(self, genotype):
    """run simulation several times"""
    # unpack genotype
    w, tau, theta, inputs_gain, motor_gain = \
        self.unpack_genotype(genotype, self.total_nodes, self.controller_inputs)

    fitnesses_q = Queue() processes = []

    # run several simulations
    for i in range(self.trials):
      orientation = np.pi * 2 / self.trials * i

      p = Process(target=self.run_simulation,
                  args=(fitnesses_q, w, tau, theta,
                        inputs_gain, motor_gain, orientation))
      p.start()
      processes.append(p)

    for p in processes:
      p.join()

    fit_std = fitnesses.std()

    # subtract 0.2 standard deviation to favour stability
    fitness = fitness_mean * stability_coeff - 0.2 * fit_std

    return fitness

  def get_random_genotype(self):
    # hidden to hidden + 3 vectors size hidden nodes: # from input to hidden
    # from hidden to output 1/2 (x2) # + 4 one to one mappings:
    # output to output 1/2 (x2) # input to output 1/2 (x2)

    if self.use_genotype_file:
      return np.array(self.loaded_genotypes.pop())

    weights_n = self.total_nodes ** 2
    tau_n = self.total_nodes
    theta_n = self.total_nodes
    inputs_gain = self.controller_inputs
    motor_gain = 1

    genes_n = weights_n + tau_n + theta_n + inputs_gain + motor_gain
    genotype = np.random.random(size=(genes_n, ))

    return genotype

  def mutate(self, value, gene_id):
    """mutation of one gene """

    # [0.01, 0.05]
    if randint(0, 1) == 0:
      mutation = 0.01 + np.random.random() * 0.04 sign = -1
    else:
      1 value += mutation * sign

    # value mast be maintained in interval [0, 1]
    # as mutation is guaranteed to be in (-1, 1)
    # we should apply the rule [0, 1](1, 0)[0, 1]...
    # this way all values [0, 1] are explored with equal probability

    if value > 1:
      rest = value % 1
      value = 1 - rest

    if value < 0:
      value = -value

    return value

  def genotype_simulation(self, genotype, interval=100, headless=False, **kwargs):
    world = World(dt=self.dt, **kwargs)

    w, tau, theta, inputs_gain, motor_gain = \
        self.unpack_genotype(np.array(genotype), self.total_nodes,
                             self.controller_inputs)

    self.contoroller = CTRNN(w, tau, theta, inputs_gain=inputs_gain,
                             motor_gain=motor_gain, total_time=self.dt,
                             time_delta=self.dt)

    poses, sensations, actions, states = world.simulate(self, interval=interval)
    fitness_1 = world.task1fitness(poses)
    fitness_2 = world.task2fitness(poses)

    print(f'Fitness 1 was: {fitness_1:.3f}')
    print(f'Fitness 2 was: {fitness_2:.3f}')
    print(f'Total fitness was: {fitness_1 + fitness_2:.3f}')

    fitness_1_full = world.task1fitness_detailed(poses)
    fitness_2_full = world.task2fitness_detailed(poses)

    print(f'fitness 1 full was: {fitness_1_full}')
    print(f'fitness 2 full was: {fitness_2_full}')

    if not headless:
      world.animate(poses, sensations)

    return fitness_1_full, fitness_2_full

  # def get_controller(self, genotype, nodes, inputs=1, total_time=0.1, time_delta=0.1):
  #   genotype = np.array(genotype)

  #   # get controller parameters from the genotype
  #   w, tau, theta, inputs_gain, motor_gain = unpack_genotype(genotype, nodes=nodes, inputs=inputs)

  #   # create a controller for a given genotype
  #   contoroller = CTRNN(w, tau, theta, inputs_gain=inputs_gain,
  #                       motor_gain=motor_gain, total_time=total_time,
  #                       time_delta=time_delta)

  #   return contoroller
