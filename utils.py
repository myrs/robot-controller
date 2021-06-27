import os
import json

import numpy as np
from scipy.stats import truncnorm


def queue_to_array(queue):
  result = []
  while not queue.empty():
    result.append(queue.get())

  return np.array(result)


def get_data(filename=None, file_index=-1):
  if not filename:
    files = os.listdir('results/')
    files.sort()
    filename = 'results/' + files[file_index]

  return json.load(open(filename))


def get_individual(filename=None, file_index=-1, run=-1, individual=-1):
  data = get_data(filename, file_index)

  individuals = data['run'][run]['individuals']
  individuals.sort(key=lambda i: i['fitness'])

  ind = individuals[individual]
  print(f'Fitness: {ind["fitness"]:.2f}')

  return ind['genotype'], ind['fitness']


def get_info(filename=None, file_index=-1):
  data = get_data(filename, file_index)

  r = {
      'settings': data['settings'],
      'fitness': data['best_solution']['fitness']}

  return r


def get_generation_genotypes(filename=None, file_index=-1, generation=-1):
  """use the last generation by default"""
  data = get_data(filename, file_index)

  return [i['genotype'] for i in data['run'][generation]['individuals']]


def angle_between(v1, v2):
  """ Returns the angle in radians between vectors 'v1' and 'v2'::

  >>> angle_between((1, 0, 0), (0, 1, 0))
  1.5707963267948966
  >>> angle_between((1, 0, 0), (1, 0, 0))
  0.0
  >>> angle_between((1, 0, 0), (-1, 0, 0))
  3.141592653589793
  """

  def unit_vector(vector):
    """ Returns the unit vector of the vector. """
    return vector / np.linalg.norm(vector)

  v1_u = unit_vector(v1) v2_u = unit_vector(v2)
  return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
