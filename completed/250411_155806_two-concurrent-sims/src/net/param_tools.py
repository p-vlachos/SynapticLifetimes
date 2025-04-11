""" Collection of utils to to create trajectories from an input_dict.

    Reimports functions from pypet, like cartesian_product, and also
    exports its own.

"""

# provide cartesian_product to users of this file
from pypet import cartesian_product


def n_list(input_dict):
    """Create explore_dict as lists instead of cartesian product

    Given an input of::

        {"a": [1.0, 2.0],
         "b": [3.0, 4.0],
         "c": [5.0]}

    it creates a explore_dict::

        {"a": [1.0, 2.0],
         "b": [3.0, 4.0],
         "c": [5.0, 5.0]}

    where columns are one experiment.
    """

    explore_dict = {}

    n = max([len(item) for key,item in input_dict.items()])

    for key,item in input_dict.items():
        if len(item) == n:
            explore_dict[key] = item
        elif len(item) == 1:
            explore_dict[key] = item*n
        else:
            raise ValueError("items must be either n or 1" +\
                             "received n=%d and value %d" %(n, len(item)))

    return explore_dict
