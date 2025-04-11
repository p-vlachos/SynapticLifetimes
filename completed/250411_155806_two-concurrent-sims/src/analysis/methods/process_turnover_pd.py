
from brian2.units import second
import sys, time
import numpy as np
import pandas as pd

def extract_lifetimes(turnover_data, N_neuron, t_cut, Tmax):
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

    parameters:
     
     -- turnover_data  :: 
     -- N_neuron       ::
     -- initial        :: determines handling of initially at 
                          t_cut present synapsese

                          - "only" :: returns lifetimes only 
                              for synapses present at t_cut
                          - "with" :: return lifetimes of 
                              synapses present at t_cut and 
                              those generated after t_cut
                          - "without" :: return lifetimes of
                              synapses generated after t_cut      
      
    returns: 

     -- lifetimes  :: duration from generation (or initial presence) 
                      until pruning. Synapses generated (initally 
                      present) but not yet pruned at simulation end 
                      are NOT INCLUDED
     -- deathtimes :: like lifetimes, but from death to generation,
                      i.e. time from begining of simulation until 
                      first generation is not included 
    '''

    # array([[  1.,   0.,   2., 347.],
    #        [  1.,   0.,   3., 248.],
    #        [  1.,   0.,   4., 145.],
    #        [  1.,   0.,  14., 210.],
    #        [  1.,   0.,  20., 318.]])
    #

    t_cut, Tmax = t_cut/second, (Tmax-t_cut)/second

    # two types of lifetimes:
    #   lts_wthsrv  -- includes synapses grown but not yet pruned
    #                  at end of simulation
    #   lts dthonly -- inlcudes only synapses grown and pruned within
    #                  simulation time              
    lts_wthsrv, lts_dthonly = [], []

    a = time.time()
    turnover_data = turnover_data[turnover_data[:,1]>= t_cut]
    turnover_data[:,1] = turnover_data[:,1]-t_cut
    b = time.time()
    print('cutting took %.2f seconds' %(b-a))


    df = pd.DataFrame(data=turnover_data, columns=['struct', 't', 'i', 'j'])

    df = df.astype({'struct': 'int64', 'i': 'int64', 'j': 'int64'})

    df['s_id'] = df['i'] * N_neuron + df['j']

    df = df.sort_values(['s_id', 't'])

    excluded_ids = []
    
    for s_id, gdf in df.groupby('s_id'):

        if len(gdf) == 1:
            if gdf['struct'].iloc[0]==1:
                # synapses started but did not die with sim tim
                # add maximal survival time t_split
                lts_wthsrv.append(Tmax - gdf['t'].iloc[0])
            else:
                # the cases are:
                #  -- synapse was present at beginning and died
                # we're not adding in this cases
                pass

        elif len(gdf) > 1:

            # we need to test that [0,1,0,1,0,1,0,...]
            # however in very rare cases it can happen that
            # [0,1,0,0,1,1,0,..]
            # we excluded these cases and count how often
            # they appear
            tar = np.abs(np.diff(gdf['struct']))

            if np.sum(tar)!=len(tar):
                excluded_ids.append(s_id)

            else:

                if gdf['struct'].iloc[0] == 0:
                    # dies first, get rid of first row
                    gdf = gdf[1:]

                if len(gdf)==1:
                    # synapses started but did not die with sim tim
                    lts_wthsrv.append(Tmax - gdf['t'].iloc[0])

                elif len(gdf)>1:

                    # starts with growth and ends on pruning event
                    # can be added to both lists
                    if len(gdf) % 2 == 0:

                        srv_t = np.diff(gdf['t'])
                        lts_wthsrv.extend(list(srv_t)[::2])
                        lts_dthonly.extend(list(srv_t)[::2])

                    # ends on growth event treat differently
                    # for lts_wthsrv and lts_dthonly
                    elif len(gdf) % 2 == 1:

                        # nothing special to do for lts_dthonly
                        srv_t = np.diff(gdf['t'])
                        lts_dthonly.extend(list(srv_t)[::2])

                        # for lts_wthsrv add above plus 
                        # extra for final surviving synapse
                        lts_wthsrv.extend(list(srv_t)[::2])
                        lts_wthsrv.append(Tmax - gdf['t'].iloc[-1])

          
       
    b = time.time()
    print('main loop took %.2f seconds' %(b-a))

    print('Excluded contacts: ', len(excluded_ids))
    
    return lts_wthsrv, lts_dthonly, excluded_ids




    
