import numpy as np
import pandas as pd

from brian2.units import second

t_split, t_cut, bin_w = 4*second, 2*second, 1*second
turnover_data = np.loadtxt('test/test_sets/turnover_test_set1',
                              delimiter=',')

df = pd.DataFrame(data=turnover_data, columns=['struct', 't', 'i', 'j'])
df = df.astype({'struct': 'int64', 'i': 'int64', 'j': 'int64'})

df['s_id'] = df['i'] * 1000 + df['j']
df = df.sort_values(['s_id', 't'])


