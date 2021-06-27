import copy
import time
import numpy as np
from scipy.stats import bradford
from multiprocessing
import Process
import Queue
from utils import queue_to_array


class MicrobialGA:
  def init(self,
           embryology,
           generations_n=100,
           individuals_n=30,
           gene_transfer_rate=0.5,
           mutation_rate=0.05,
           replace_rate=0,
           ranking_level=0,
           run_parallel=False):

    self.calculate_fitness = embryology.calculate_fitness
    self.get_random_genotype = embryology.get_random_genotype
    self.mutate = embryology.mutate
    self.embryology = embryology
    self.individuals_n = individuals_n
    self.generations_n = generations_n
    self.gene_transfer_rate = gene_transfer_rate
    self.mutation_rate = mutation_rate
    self.run_parallel = run_parallel
    self.replace_rate = replace_rate
    self.ranking_level = ranking_level

    self.best_individual_fintesses = []
    self.best_historical_fintess = -np.inf
    self.generation_number = 0
    self.generations_data = []

  def initialize_population(self):
    """
    initialized population as a dictionary,
    where each individual obtains a random a genotype and fitness set to None
    """
    self.population = []

    for _ in range(self.individuals_n):
      genotype = self.get_random_genotype()
      fitness = self.calculate_fitness(genotype)

      individual = {'genotype': genotype, 'fitness': fitness}

      self.population.append(individual)

    self.population = np.array(self.population)

    # store generations data for the first generation
    self.store_generation_data()

  def store_generation_data(self):
    generation_data = {
        'number': self.generation_number,
        'total_fitness': 0,
        'best_individual_genotype': None,
        'best_individual_fitness': None,
        'individuals': []
    }

    best_individual = self.population[0]

    for individual in self.population:
      generation_data['individuals'].append(copy.deepcopy(individual))

      # update the best fit individual
      if individual['fitness'] > best_individual['fitness']:
        best_individual = individual

    generation_data['best_individual_genotype'] = best_individual['genotype']
    generation_data['best_individual_fitness'] = best_individual['fitness']

    self.best_individual_fintesses.append(best_individual['fitness'])
    self.best_historical_fintess = best_individual['fitness']

    self.generations_data.append(generation_data)

  # prevent from competing individual with itself
  def get_random_individual(self, resticted_id=None, min_id=0):
    """
    returns random individual from population if %restricted_id is provided
    ensures selected individual in not the same, that restricted one if not, executes recursively
    """
    # prioritize good solutions to compete
    if self.ranking_level:
      last_id = len(self.population) - 1
      length = last_id - min_id
      dist = bradford(3, min_id, length)
      individual_id = int(dist.rvs())
      individual_id = last_id - individual_id + min_id

    else:
      individual_id = np.random.randint(min_id, len(self.population))

    if resticted_id is not None and individual_id == resticted_id:
      return self.get_random_individual(resticted_id)

    return self.population[individual_id], individual_id

  def get_winner_and_looser(self, individual_1, individual_2):
    # compare fitness in a probabilistic manner
    # because it's just and estimation and not the real value

    # fitness_diff = individual_1['fitness'] - individual_2['fitness']

    # # compare with 0.5 standard deviation
    # # 0.5 is derived from tests, that were run
    # prob = np.random.normal(loc=fitness_diff, scale=0.2)

    # if prob > 0:

    # return individual_1, individual_2

    if individual_1['fitness'] > individual_2['fitness']:
      return individual_1, individual_2

    return individual_2, individual_1

  def microbial_sex(self, winner, loser):
    for i, _ in enumerate(winner['genotype']):
      if self.gene_transfer_rate > np.random.random():
        loser['genotype'][i] = winner['genotype'][i]

      if self.mutation_rate > np.random.random():
        loser['genotype'][i] = self.mutate(loser['genotype'][i], i)

  def round(self, new_popultaion_q, individual_1, individual_2):
    # reset random seed
    np.random.seed()

    individual_1 = copy.deepcopy(individual_1)
    individual_2 = copy.deepcopy(individual_2)

    winner, loser = self.get_winner_and_looser(individual_1, individual_2)
    # copy the genes of the winner to loser with %gene_transfer_rate
    # and mutate looser genes with %mutation_rate
    self.microbial_sex(winner, loser)

    # update fitness function for looser
    loser['fitness'] = self.calculate_fitness(loser['genotype'])

    new_popultaion_q.put(loser)
    new_popultaion_q.put(winner)

  def tournament_selection_parallel(self):
    # new population queue
    new_popultaion_q = Queue()
    processes = []

    np.random.shuffle(self.population)

    # repeat number of individuals / 2 times
    for i in range(int(self.individuals_n / 2)):
      individual_1 = self.population[i * 2]
      individual_2 = self.population[i * 2 + 1]
      p = Process(target=self.round, args=(new_popultaion_q, individual_1, individual_2))
      p.start()
      processes.append(p)

    for p in processes:
      p.join()

    self.population = queue_to_array(new_popultaion_q)

  def tournament_selection(self):
    # repeat %length_of_population times divided by 2
    for i in range(int(self.individuals_n / 2)):
      # select two individuals
      individual_1, individual_id_1 = self.get_random_individual()
      individual_2, _ = self.get_random_individual(individual_id_1)

      winner, loser = self.get_winner_and_looser(individual_1, individual_2)
      # copy the genes of the winner to loser with %gene_transfer_rate
      # and mutate looser genes with %mutation_rate
      self.microbial_sex(winner, loser)

      # update fitness function for looser
      loser['fitness'] = self.calculate_fitness(loser['genotype'])

  def replace_poor(self):
    if not self.replace_rate:
      return

    # remove bottom replace_rate solutions
    # population = copy.deepcopy(self.population)
    self.population = sorted(self.population, key=lambda x: x['fitness'])

    last_poor = int(self.individuals_n * self.replace_rate)
    first_top = last_poor + 1

    for i in range(int(self.individuals_n * self.replace_rate)):
      ind, _ = self.get_random_individual(min_id=first_top)
      ind = copy.deepcopy(ind)

      for j, _ in enumerate(ind['genotype']):
        if self.mutation_rate > np.random.random():
          ind['genotype'][j] = self.mutate(ind['genotype'][j], j)

        ind['fitness'] = self.calculate_fitness(ind['genotype'])
        self.population[i] = ind

  def run(self):
    # initialize population
    self.initialize_population()

    self.start_time = time.time()
    while self.generation_number < self.generations_n:
      start = time.time()

      self.generation_number += 1
      # uncomment for printing current generation
      print(f'\nGeneration {self.generation_number} of {self.generations_n}...')

      if self.run_parallel:
        self.tournament_selection_parallel()
      else:
        self.tournament_selection()

      self.replace_poor()
      self.store_generation_data()

      print(f'Best fitness {self.best_historical_fintess}')
      print(f'Evaluation time: {time.time() - start}')

    print(f'\ntotal time: {time.time() - self.start_time:.2f}\n')
