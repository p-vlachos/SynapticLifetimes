
import numpy as np
from brian2.units import second


def get_insert_and_prune_counts(turnover_data, N_neuron, t_cut, Tmax):
    '''
    turnover data is assumed to be a numpy.array with 
    lines consisting of the four entries
    
      gen/prune, t, i, j     

    where 
      -- gen/prune :: 1 if synapse became active, 
                      0 if synapse became inactive
      -- t         :: simulation time point in seconds(!)
      -- i         :: pre-synaptic neuron index
      -- j         :: post-synaptic neuron index

    -----------

    returns:
    
    ...

    '''

    t_cut, Tmax = t_cut/second, Tmax/second

    turnover_data = turnover_data[turnover_data[:,1]<= Tmax]
    turnover_data = turnover_data[turnover_data[:,1]>= t_cut]
    
    insert_events = turnover_data[turnover_data[:,0] == 1][:,3]
    prune_events  = turnover_data[turnover_data[:,0] == 0][:,3]
  
    bins = np.arange(0, N_neuron+1, 1)

    assert(np.max(insert_events) <= N_neuron)
    assert(np.max(prune_events) <= N_neuron)

    insert_counts, bins =np.histogram(insert_events,
                                      bins=bins, density=False)

    prune_counts, bins =np.histogram(prune_events,
                                      bins=bins, density=False)

    return insert_counts, prune_counts


 
