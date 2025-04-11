from net.analysis_cache import CachedAnalysis


def operate_by_regime(directories, bin_w, operation):
    out_subcrit, out_revreg = [], []
    for dir in directories:
        with CachedAnalysis(dir) as cache:
            fit_mre = cache.get_mre(bin_w)

            if fit_mre > 0.995:
                print(f"skipping: mre is critical or super-critical: {dir}")
                continue
            out = operation(dir)
            if fit_mre > 0.9:
                out_revreg.append(out)
            else:
                out_subcrit.append(out)
    return [out_subcrit, out_revreg]
