# using secrets for fun
import secrets 
import multiprocessing as mp
import numpy as np

def coin_flip() -> int: return secrets.choice([0, 1]) # 'heads' or 'tails'

def bernoulli_coin_flip(num_flips:int = 100) -> int:
  return sum([coin_flip() for _ in range(num_flips)])

def lot_bernoulli_coin_flip(num_flips:int = 100, num_exps:int =100) -> list[int]:
  """creates `num_exps` of bernoulli coin flips and returns num of `tails` 
  * `heads = numm_flips-tails`

  Args:
      num_flips (int, optional): times of toss the coin. Defaults to 100.
      num_exps (int, optional): times of run eperiment of toss the coin `num_flips` times. Defaults to 100.

  Returns:
      list[int]: list with count of tails in  the experiment.
  """
  with mp.Pool(processes=16) as pool:
        results = pool.map(bernoulli_coin_flip, [num_flips]*num_exps)
  return np.array(results)


