
import numpy as np

def convert_to_EE_adjacency_matrix(syn_array, N):

    assert np.shape(syn_array) == tuple([N*(N-1)]), \
        "Synpase array doesn't have the expected shape"

    # construct i,j arrays
    i = np.repeat(range(N), N)
    j = np.tile(range(N), N)

    idx = (i!=j)
    i,j = i[idx], j[idx]

    adj_matrix = np.zeros((N,N))

    for syn_el, row_id, col_id in zip(syn_array,i,j):
        adj_matrix[row_id, col_id] = syn_el

    return adj_matrix


def convert_to_EI_adjacency_matrix(syn_array, N_E, N_I):

    assert np.shape(syn_array) == tuple([N_I*N_E]), \
        "Synpase array doesn't have the expected shape"

    # construct i,j arrays
    i = np.repeat(range(N_I), N_E)
    j = np.tile(range(N_E), N_I)

    adj_matrix = np.zeros((N_I,N_E))

    for syn_el, row_id, col_id in zip(syn_array,i,j):
        adj_matrix[row_id, col_id] = syn_el

    return adj_matrix

        

def topright_axis_off(ax):

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')


if __name__ == "__main__":
    
    convert_to_adjacency_matrix([0,1], 2)
