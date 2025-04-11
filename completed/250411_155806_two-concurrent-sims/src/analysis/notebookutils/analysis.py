import numpy as np
import h5py
from brian2 import asarray
from brian2.units import *
import pickle


def index_to_build(index):
    return f"{index:0>4d}"


def get_attr(f: h5py.File, index, attr):
    keyname, valuekey = None, None
    for keyname_candidate, valuekey_candiate in [("explored_data__brn2__", "value"), ("explored_data", "data")]:
        if keyname_candidate in f["tr1"]["parameters"]["netw"][attr]:
            keyname = keyname_candidate
            valuekey = valuekey_candiate
    return f["tr1"]["parameters"]["netw"][attr][keyname][index][0]


def get_attrs(f: h5py.File, index, attr_query):
    return {attr: get_attr(f, index, attr) for attr in attr_query}


class Analysis(object):

    def __init__(self, sim_paths, names=None):
        self._paths = sim_paths
        self._names = names
        if self._names is None:
            self._names = self._paths

    def get_builds(self, query, attr_query):
        builds = []
        for name, path in zip(self._names, self._paths):
            f = h5py.File(f"{path}/data/hdf5_data.hdf5", "r")
            for key, value in query.items():
                keyname, valuekey = None, None
                for keyname_candidate, valuekey_candiate in [("explored_data__brn2__", "value"), ("explored_data", "data")]:
                    if keyname_candidate in f["tr1"]["parameters"]["netw"][key]:
                        keyname = keyname_candidate
                        valuekey = valuekey_candiate
                indexs = np.where(f["tr1"]["parameters"]["netw"][key][keyname][valuekey] == asarray(value))[0]
                builds += [(path,
                            index_to_build(index),
                            get_attrs(f, index, attr_query)) for index in indexs]
                builds[-1][2].update(dict(sim_name=name))
        return builds

    def get_build(self, id, path=None, attr_query=None):
        path = self._paths[0] if path is None else path
        name = None
        for path_, name_ in zip(self._paths, self._names):
            if path_ == path:
                name = name_
                break
        f = h5py.File(f"{path}/data/hdf5_data.hdf5", "r")
        build = (path, id, get_attrs(f, int(id), attr_query))
        build[2].update(dict(sim_name=name))
        return build

    def get_hdfresult(self, build, key):
        f = h5py.File(f"{build[0]}/data/hdf5_data.hdf5", "r")
        return f['tr1']['results']['runs'][f'run_0000{build[1]}'][key][key]

    def get_bpath(self, build):
        return f"{build[0]}/builds/{build[1]}"

    def get_nsp(self, build):
        return load_nsp(build)


def pickle_load(build, file, n=1, print_attrs=True):
    ret = []
    with open(f"{build[0]}/builds/{build[1]}/raw/{file}.p", "rb") as f:
        for _ in range(n):
            ret.append(pickle.load(f))
    if n == 1:
        if print_attrs:
            print(file, ret[0].keys(), sep=": ")
        return ret[0]
    else:
        return ret


def load_nsp(build):
    return pickle_load(build, "namespace", print_attrs=False)


def find_bounds(data):
    """index_bounds is [start:end], but time_bounds is [start, ..., end], i.e. index_bounds can be used with Python
       indexing but time_bounds' upper bound is the actual upper bound, not the next element.
    """
    ts = data['t']
    tdiffs = np.diff(ts)
    count, bins = np.histogram(tdiffs)
    dt_measure = np.min(bins[:-1][count > 0]) * second
    boundaries = np.hstack(((tdiffs > 2 * dt_measure).nonzero()[0], -1))
    index_bounds = np.vstack([np.roll(boundaries + 1, 1), boundaries]).T
    time_bounds = ts[index_bounds]
    index_bounds[index_bounds == -1] = len(ts)-1
    index_bounds[:, 1] += 1
    return index_bounds, time_bounds


def get_label(build, label_attr=None):
    attrs_ = build[2]
    if label_attr == "build_index":
        return build[1]
    if len(attrs_) == 0:
        return ""
    def rename_key(key):
        if key == "random_seed":
            return "seed"
        elif key == "strong_mem_noise_rate":
            return "inp_rate"
        return key
    attrs = {rename_key(key): value for key, value in attrs_.items() if label_attr is None or rename_key(key) in label_attr}
    if "build_index" in label_attr:
        attrs['build_index'] = build[1]
    return ",".join([f"{key}={value}" for key, value in attrs.items()])
