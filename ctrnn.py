from matplotlib import pyplot as plt
import numpy as np


def sigmoid(y, theta):
  # add both default bias (theta) and hunger bias H
  return 1 / (1 + np.exp(-(y + theta)))


class CTRNN:
  def __init__(self, w, tau, theta,
               inputs_gain=[1, 1, 1, 1], motor_gain=1,
               total_time=0.1, time_delta=0.1):

    self.total_time = total_time
    self.dt = time_delta
    self.w = w
    self.tau = tau
    self.theta = theta
    self.n_nodes = w.shape[0]
    self.y = [np.zeros((self.n_nodes, 1))]
    self.inputs_gain = inputs_gain
    self.motor_gain = motor_gain

  def output(self, input_1=0, input_2=0, input_3=0, input_4=0):
    iterations = int(self.total_time / self.dt)

    # sensor is only for the first neuron
    I = np.zeros((self.n_nodes, 1))

    # normal light sensor
    I[0, 0] = input_1 * self.inputs_gain[0]
    # memory light sensor
    I[3, 0] = input_2 * self.inputs_gain[1]
    # motor 1 sensor
    I[4, 0] = input_3 * self.inputs_gain[2]
    # motor 2 sensor
    I[5, 0] = input_4 * self.inputs_gain[3]

    for i in range(0, iterations):
      self.iterate(I)
    
    yt = self.y[-1]
    
    # motor neurons are the 2nd and 3rd
    motor_out = self.motor_gain * np.tanh(yt[[1, 2], 0])

    return motor_out, None

  def iterate(self, I):
    yt = self.y[-1]

    yt1 = yt + self.dt * (1 / self.tau) * (-yt + self.w @ sigmoid(yt, self.theta) + I)
    
    self.y.append(yt1)

  def show(self):
    # plt.ion()
    plt.figure()

    time = range(0, len(self.y))
    y = np.array(self.y)
    y = y.reshape(y.shape[0], y.shape[1])
    y = y[:, [0, -2, -1]]
    
    # need to transpose y, so it would have the same shape as time
    plt.plot(time, y, linewidth=0.75)
    plt.show(block=True)




