import h5py
from os import path
import numpy as np

from brian2 import Quantity


def _normalize_item(item):
    if type(item) is str:
        return item
    elif type(item) is Quantity:
        return str(item)  # we rely on this being consistent
    return str(item)


def _normalize(group_name_path: list):
    return "/".join([_normalize_item(item) for item in group_name_path])


class CacheRef:

    def __init__(self, cache: h5py.File, ds_name: str, build_dir):
        self.cache = cache
        self.ds_name = ds_name
        self.build_dir = build_dir

    def exists(self):
        return self.ds_name in self.cache

    def data(self):
        ds = self.cache[self.ds_name]
        if ds.shape == ():  # scalar
            return ds[()]
        return np.copy(ds)

    def store(self, data):
        return self.cache.create_dataset(self.ds_name, data=data, compression='gzip', compression_opts=9)

    def store_scalar(self, data):
        return self.cache.create_dataset(self.ds_name, data=data)

    def retrieve(self, calc_f):
        if self.exists():
            return self.data()
        else:
            res = calc_f(self.build_dir)
            try:
                _ = res.shape
            except AttributeError:
                is_scalar = True
            else:
                if res.shape == ():
                    is_scalar = True
                else:
                    is_scalar = False
            if is_scalar:
                self.store_scalar(res)
            else:
                self.store(res)
            return res


class AnalysisCache:

    def __init__(self, build_dir, ds_path=None):
        raw_dir = path.join(build_dir, 'raw')
        if not path.exists(raw_dir):
            raise FileNotFoundError(f"Could not find {raw_dir}")
        self.build_dir = build_dir
        self.sim_dir = path.dirname(path.dirname(build_dir))  # directory/builds/0000 -> directory/
        self.build_name = path.basename(build_dir)
        self.cache_file = path.join(self.sim_dir, "analysis_cache.hdf5")  # directory/analysis_cache/0000.hdf5

        self.cache = h5py.File(self.cache_file, mode='a')
        self.ds_path = ds_path

    def cache_ref(self, path: list):
        ds_name = _normalize([self.build_name] + path)
        return CacheRef(self.cache, ds_name, self.build_dir)

    def __enter__(self):
        if self.ds_path is None:
            return self
        else:
            return self.cache_ref(self.ds_path)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cache.close()


class CachedAnalysis:

    def __init__(self, build_dir):
        self.cache = AnalysisCache(build_dir)
        with open(build_dir + '/raw/namespace.p', 'rb') as pfile:
            import pickle
            self.nsp = pickle.load(pfile)

    def _get_or_calc(self, path, calc_f):
        return self.cache.cache_ref(path).retrieve(calc_f)

    def get_mre(self, bin_w):
        import mrestimator as mre
        from analysis.axes.network import branching_ratio

        def calc_mre(build_dir):
            # we assume that selected builds pass stationary tests
            rk, ft, _ = branching_ratio('', build_dir, self.nsp, bin_w)
            fit = mre.fit(rk, fitfunc=mre.f_exponential_offset)
            return fit.mre

        return self._get_or_calc(["branching_ratio", bin_w], calc_mre)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cache.cache.close()
