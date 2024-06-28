import numpy as np
import matplotlib.pyplot as plt

data_base = np.loadtxt('MPLTS/Scripts/base.data')
data_opt = np.loadtxt('MPLTS/Scripts/opt.data')

cost_base = data_base[:,4]
cost_opt = data_opt[:,4]

bestcost_base = [min(cost_base[:i+1]) for i in range(len(cost_base))]
bestcost_opt = [min(cost_opt[:i+1]) for i in range(len(cost_opt))]

plt.figure(figsize=(10, 6))
plt.plot(bestcost_base, label='base')
plt.plot(bestcost_opt, label='opt')

plt.xlabel('searching graphs')
plt.ylabel('opt % of origin graph')
plt.legend()

plt.savefig('MPLTS/Scripts/prune.png')
plt.close()