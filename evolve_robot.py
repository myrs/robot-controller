import json
from datetime import datetime
from numpyencoder import NumpyEncoder

from robot_embryology import RobotEmbryology
from microbial_ga import MicrobialGA


def evolve_robot(controller_1_file=None,
                 generations_n=10,
                 individuals_n=6,
                 gene_transfer_rate=0.5,
                 mutation_rate=0.05,
                 save_to_file=True,
                 store_only_best=False,
                 interval=100,
                 dt=0.1,
                 hidden_nodes=8,
                 min_stability=None,
                 run_parallel=False,
                 replace_rate=0,
                 ranking_level=0,
                 use_genotype_file=None,
                 trials=12,
                 ignore_inputs=False,
                 world_params={}):

  embryology = RobotEmbryology(controller_1_file=controller_1_file,
                               dt=dt,
                               hidden_nodes=hidden_nodes,
                               interval=interval,
                               min_stability=min_stability,
                               use_genotype_file=use_genotype_file,
                               world_params=world_params,
                               trials=trials,
                               ignore_inputs=ignore_inputs)

  ga = MicrobialGA(embryology,
                   generations_n=generations_n,
                   individuals_n=individuals_n,
                   gene_transfer_rate=gene_transfer_rate,
                   mutation_rate=mutation_rate,
                   run_parallel=run_parallel,
                   replace_rate=replace_rate,
                   ranking_level=ranking_level)

  ga.run()
  data = ga.generations_data
  best_solution_fitness = data[-1]['best_individual_fitness']
  best_solution_genotype = data[-1]['best_individual_genotype']
  print('Best solution was:')
  print(best_solution_fitness)
  print('Genotype:')
  print(best_solution_genotype)

  data = {
      'best_solution': {
          'fitness': best_solution_fitness,
          'genotype': best_solution_genotype
      },
      'run': data,
      'settings': {
          'ga': {
              'generations': generations_n,
              'population': individuals_n,
              'gene_transfer_rate': gene_transfer_rate,
              'mutation_rate': mutation_rate,
              'run_parallel': run_parallel,
              'replace_rate': replace_rate,
              'ranking_level': ranking_level,
          },
          'embryology': {
              'controller_1_file': controller_1_file,
              'hidden_nodes': hidden_nodes,
              'min_stability': min_stability,
              'interval': interval,
              'hunger': 'yes',
              'both': 'yes',
              'trials': trials,
              'noise_variation': False,
              'ignore_inputs': ignore_inputs,
          }
      },
  }

  if save_to_file:
    timestamp = datetime.strftime(datetime.now(), '%y_%m_%d__%H_%M_%S')
    with open(f'results/run__{timestamp}.json', 'w') as file:
      json.dump(data, file, cls=NumpyEncoder)

    with open(f'results/best__{timestamp}.json', 'w') as file:
      json.dump(data['best_solution'], file, cls=NumpyEncoder)

  return data, best_solution_genotype, embryology
