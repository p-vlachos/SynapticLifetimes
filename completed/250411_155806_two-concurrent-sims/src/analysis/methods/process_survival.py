import brian2.units
from brian2.units import second
import sys, time
import numpy as np
import pandas as pd


def extract_survival(turnover_data: np.ndarray, N_neuron: int, t_split, t_cut: brian2.units.Quantity = 0. * second):
    """ Extract synapse survival times from raw turnover data

    :param turnover_data: lines of [1 gen/0 prune, t (seconds),
        i (pre-synaptic neuron), j (post-synaptic neuron)]
    :param N_neuron: exc. neurons in network
    :param t_split: Duration of window from first growth event in which other growth events are considered.
    :param t_cut: Only synapse events after this time are considered.

    Events that we consider:

        * synapse was created within t_split but did not die within sim time -> t_split
        * growth within t_split and pruning is within sim time -> actual dt (maximum is t_split)

    Events we ignore:

        * synapse was created at beginning and died later
        * synapse was created after t_split
        * synapse has two creation events without intermittent destruction -> ``excluded_ids``

    :returns: (survival duration in seconds, excluded_ids)
    """

    t_split, t_cut = t_split / second, t_cut / second

    # start at offset t_cut
    a = time.time()
    turnover_data = turnover_data[turnover_data[:, 1] >= t_cut]
    turnover_data[:, 1] = turnover_data[:, 1] - t_cut
    b = time.time()
    print('cutting took %.2f seconds' % (b - a))

    # construct pandas dataframe
    df = pd.DataFrame(data=turnover_data, columns=['struct', 't', 'i', 'j'])
    df = df.astype({'struct': 'int32', 'i': 'int32', 'j': 'int32'})
    df['s_id'] = df['i'] * N_neuron + df['j']
    df = df.sort_values(['s_id', 't'])
    print('got pandas frame')

    full_t = []
    excluded_ids = []
    for s_id, gdf in df.groupby('s_id'):

        if len(gdf) == 1:
            if gdf['struct'].iloc[0] == 1 and gdf['t'].iloc[0] <= t_split:
                # synapses grew but did not die within sim time
                # add maximal survival time t_split
                full_t.append(t_split)
            else:
                # the cases are:
                #  -- synapse was present at beginning and died
                #  -- synapse grew but we don't have enough time
                #     data to track it's full survival
                #
                # we're not adding in these cases
                pass

        elif len(gdf) > 1:
            # we need to test that [0,1,0,1,0,1,0,...]
            # in very rare cases it can happen that [0,1,0,0,1,1,0,..]
            # we excluded these cases and count how often they appear
            # TODO WHY does this happen ?????
            tar = np.abs(np.diff(gdf['struct']))
            if np.sum(tar) != len(tar):
                excluded_ids.append(s_id)
                continue  # abort processing of this synapse

            if gdf['struct'].iloc[0] == 0:
                # dies first, get rid of first row
                gdf = gdf[1:]

            if len(gdf) == 1 and gdf['t'].iloc[0] <= t_split:
                # synapses started but did not die with sim time: add maximal survival time t_split
                full_t.append(t_split)
            elif len(gdf) > 1 and gdf['t'].iloc[0] <= t_split:
                # consider only if first growth event is before t_split, otherwise there is not enough time to track
                # potentially long surviving synapses

                # normalize to the times of the first growth event
                # +gdf['t'] = gdf['t']-gdf['t'].iloc[0]+
                # --> done below by adding to t_split

                # filter out events after window t_split
                gdf_cut = gdf[gdf['t'] <= t_split + gdf['t'].iloc[0]]

                if len(gdf_cut) % 2 == 0:
                    # starts with growth event, ends with pruning event
                    # (cannot start on pruning event since we have removed leading pruning events earlier)
                    srv_t = np.diff(gdf_cut['t'])
                    assert np.max(srv_t) <= t_split
                    full_t.extend(list(srv_t)[::2])
                elif len(gdf_cut) % 2 == 1:
                    # starts with growth event, ends with growth event => need to find next pruning event
                    if len(gdf_cut) == 1:
                        if len(gdf_cut) == len(gdf):
                            # we didn't filter any events out by t_split
                            # therefore can't find any pruning event, add maximal survival
                            full_t.append(t_split)
                        else:
                            # we filtered events out by t_split
                            # the next element has to be a pruning event since this element is a growth event
                            # and since in this branch len(gdf_cut) == 1, len(gdf_cut) is the index of the next event
                            dt = gdf['t'].iloc[len(gdf_cut)] - gdf_cut['t'].iloc[0]
                            if dt > t_split:
                                # TODO this appears unnecessary, if we filtered out this event definitely dt > t_split
                                full_t.append(t_split)
                            else:
                                full_t.append(dt)
                    elif len(gdf_cut) > 1:
                        # (still) starts with growth event, (still) ends with growth event
                        gdf_blk, gdf_end = gdf_cut[:-1], gdf_cut.iloc[-1]
                        assert (gdf_blk['struct'].iloc[0] == 1)
                        assert (gdf_end['struct'] == 1)
                        # process events (all in t_split) except last growth (need to look for prune outside of t_split)
                        srv_t = np.diff(gdf_blk['t'])
                        assert np.max(srv_t) <= t_split
                        full_t.extend(list(srv_t)[::2])
                        # process the final growth event gdf_cut
                        if len(gdf_cut) == len(gdf):
                            # final growth event within t_split did not get pruned until simulation end, add max
                            full_t.append(t_split)
                        elif len(gdf_cut) < len(gdf):
                            dt = gdf['t'].iloc[len(gdf_cut)] - gdf_end['t']
                            if dt > t_split:
                                # TODO similar argument as above
                                full_t.append(t_split)
                            else:
                                full_t.append(dt)

    b = time.time()
    print('main loop took %.2f seconds' % (b - a))

    print('Excluded contacts: ', len(excluded_ids))

    return full_t, excluded_ids


def convert_full_t_to_srv_prb(full_t, t_max, bin_w):
    bins = np.arange(bin_w, t_max + bin_w, bin_w)

    counts, edges = np.histogram(full, bins=bins, density=False)

    np.cumsum(counts[::-1])[::-1]
