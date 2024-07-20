import bernoulli  
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from collections import Counter
from functools import reduce 
from fitter import Fitter

if __name__ == "__main__":
  num_exps = 100000
  num_flips = 1000
  exp_result = np.load("cached_result.npy") if Path("cached_result.npy").exists() else bernoulli.lot_bernoulli_coin_flip(num_flips,num_exps) 
  if not Path("cached_result.npy").exists():
    np.save("cached_result.npy", exp_result)
  result_log = Path("result_log.log").open('w')
  # result_log.write(f"{exp_result =}\n")
  result_log.write(f"Num of exps {num_exps} num_flips {num_flips}\n")
  result_log.write(f"mean {np.mean(exp_result)} vs median {np.percentile(exp_result, q=50)}\n")
  result_log.write(f"{np.percentile(exp_result, q=1) = }\n")
  result_log.write(f"{np.percentile(exp_result, q=99) = }\n")
  result_log.write(f"σ = {np.std(exp_result)}\n")

  counted = Counter(exp_result)
  result_log.write(f"actual chance to pass into 3σ is {100-100*reduce(lambda x, y: x + y[1], counted.most_common()[round(6*np.std(exp_result)):], 0)/num_exps}% vs 99.73% from theory\n")
  result_log.write(f"actual chance to pass into 2σ is {100-100*reduce(lambda x, y: x + y[1], counted.most_common()[round(4*np.std(exp_result)):], 0)/num_exps}% vs 95.45% from theory\n")
  result_log.write(f"actual chance to pass into 1σ is {100-100*reduce(lambda x, y: x + y[1], counted.most_common()[round(2*np.std(exp_result)):], 0)/num_exps}% vs 68.27% from theory\n")
  result_log.write(f"\nDist explore:\n")
  fitter = Fitter((exp_result - np.mean(exp_result))/np.sqrt(np.std(exp_result) + 1e-6), distributions=['norm', 'expon', 'gamma', 'lognorm', 'beta'])
  fitter.fit(progress=True, n_jobs=17)
  result_log.write(f"{fitter.summary()}\n")
  best_distribution = fitter.get_best(method='sumsquare_error')
  result_log.write(f"Best distribution: {best_distribution}")
  
  
  
  
  result_log.close()
  sns.displot(exp_result)
  plt.savefig("dist_dist_coinflip.png")
  
  print(exp_result.shape)