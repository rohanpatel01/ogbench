
import numpy as np

# data = np.load('/work/10993/rohanpatel01/vista/ogbench/impls/data_gen_scripts/data/intermediate_save_500/visual-antmaze-medium-stitch-v0-distracted.npz')

data = np.load('/work/10993/rohanpatel01/vista/ogbench/impls/data_gen_scripts/data/intermediate_save_500/visual-antmaze-medium-stitch-v0-distracted-val.npz')
print(data['terminals'].dtype)       # should be bool
print(data['terminals'].sum())       # should be > 0
print(data['terminals'][-1])         # should be True