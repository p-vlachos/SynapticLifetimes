
import numpy as np
import pandas as pd
from brian2.units import second



def resample_spk_register(spkr, tr, resamp_dt):

    
    df = pd.DataFrame(data=spkr, columns=['t', 'i', 'j', 'a', 'Apre',
                                          'Apost', 'preorpost'])

    df = df.astype({'preorpost': 'int64', 'i': 'int64', 'j': 'int64'})


    df['s_id'] = df['i'] * tr.N_e + df['j']


    # structure of data is
    #
    #         a          Apre         Apost  preorpost  
    #  0.006074  6.819270e-25 -7.500000e-05          1  
    #  0.006072  1.154220e-29 -7.500000e-05          1  
    #  0.006072  1.500000e-04 -7.782750e-09          0  
    #  0.006072  1.500000e-04 -8.061340e-11          0  
    #  0.006072  1.432580e-12 -7.500000e-05          1
    #
    # introduce new variable 'dA' which is the weight change
    # agnostic of whether it comes from pre or post.

    df['dA'] = df['preorpost'] * df['Apre'] + (1-df['preorpost'])*df['Apost']

    # sort by neuron pair and time
    df = df.sort_values(['s_id', 't'])

    # normalize to T=0
    df['t'] = df['t'] - (tr.T1+tr.T2+tr.T3)/second


    bin_w = resamp_dt/second
    T = tr.stdp_rec_T/second
    
    dt_vals = list(np.arange(0,T,bin_w))
    dA_vals = []
    a_vals = []

    for s_id, gdf in df.groupby('s_id'):

        dA_sum, a_init = [], []
    
        for s in dt_vals:

            view = gdf[(gdf['t'] >= s) & (gdf['t']<s+bin_w)]

            dA_sum.append(np.sum(view['dA']))

            try:
                a=view['a'].iloc[0]
            except IndexError:
                # not a single weight change was recorded in
                # in the bin_w interval.
                # most likely, the synapse grew/died during T
                # and was not present yet/anymore during the
                # current (s,s+bin_w) period. mark with -1
                # and discard later
                a=-1
            a_init.append(a)

        dA_vals.append(dA_sum)
        a_vals.append(a_init)


    return np.array(dA_vals), np.array(a_vals)

            
    

def resample_scl_deltas(scl_deltas, tr, resamp_dt):

    df = pd.DataFrame(data=scl_deltas,
                      columns=['t', 'a', 'a_scl', 'i', 'j'])
    df = df.astype({'i': 'int64', 'j': 'int64'})

    df['s_id'] = df['i'] * tr.N_e + df['j']
    df['dA'] = df['a_scl'] - df['a']

    # sort by neuron pair and time
    df = df.sort_values(['s_id', 't'])

    # normalize to T=0
    df['t'] = df['t'] - (tr.T1+tr.T2+tr.T3)/second


    bin_w = resamp_dt/second
    T = tr.scl_rec_T/second

    dt_vals = list(np.arange(0,T,bin_w))
    dA_vals = []
    a_vals = []

    for s_id, gdf in df.groupby('s_id'):

        dA_sum, a_init = [], []

        for s in dt_vals:

            view = gdf[(gdf['t'] >= s) & (gdf['t']<s+bin_w)]

            dA_sum.append(np.sum(view['dA']))

            try:
                a=view['a'].iloc[0]
            except IndexError:
                # not a single weight change was recorded in
                # in the bin_w interval.
                # most likely, the synapse grew/died during T
                # and was not present yet/anymore during the
                # current (s,s+bin_w) period. mark with -1
                # and discard later
                a=-1
            a_init.append(a)

        dA_vals.append(dA_sum)
        a_vals.append(a_init)


    return np.array(dA_vals), np.array(a_vals)





