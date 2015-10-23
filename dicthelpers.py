import numpy as np

def attrfilter(d, indexed=False, **attr):
    test = lambda x: all(getattr(x, k, None) == v for k, v in attr.iteritems())
    if isinstance(d, dict):
        d = d.itervalues()
    if indexed:
        vs = []
        indices = []
        for i, x in enumerate(d):
            if test(x):
                vs.append(x)
                indices.append(i)
        return vs, indices
    else:
        return filter(test, d)

def attrdata(d, k, s=None):
    if s is None: s = slice(None)
    return np.asarray([getattr(v, k)[s] for v in d.itervalues()])
