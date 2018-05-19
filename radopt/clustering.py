import functools
import collections
import numpy as np

class DiscreteMap(object):
    pass

class IdentityMap(DiscreteMap):
    def downsample(self, array, **options):
        return 1. * array

    def upsample(self, array, **options):
        return 1. * array

class ClusterMap(DiscreteMap):
    """ Wrap a vector of point->cluster assignments as a linear operator

    """
    def __init__(self, n_points, n_clusters, assignments):
        DiscreteMap.__init__(self)
        assert n_points >= n_clusters, "# points >= # clusters"
        assert len(assignments) == n_points, "# points = len(assignments)"
        assert np.max(assignments) <= n_clusters - 1, "max assignment <= # clusters"
        self.n_points = int(n_points)
        self.n_clusters = int(n_clusters)
        self.vec = np.array(assignments).astype(int)
        self.cluster_weights = np.zeros(n_clusters)
        for a in assignments:
            self.cluster_weights[a] += 1

    @property
    def assignments(self):
        return self.vec

    def downsample(self, array, normalize=True, **options):
        r"""
        Given A in R^{n_points \times n}, form C = U^TA in R^{n_clusters \times n}

        If `normalize` is ``True``, form C = (U^TU)^{-1}U^TA to normalize
        the rows of the output in the clustered space by the number of
        points in each cluster.
        """
        orient = np.transpose if options.get('transpose', False) else lambda A: A
        array = orient(array)
        shape = list(array.shape)
        shape[0] = self.n_clusters
        output = np.zeros(shape)
        for idx_pts, idx_clu in enumerate(self.assignments):
            output[idx_clu, ...] += array[idx_pts, ...]

        if normalize:
            for i, w in enumerate(self.cluster_weights):
                if w > 0:
                    output[i, ...] /= w

        return orient(output)

    def upsample(self, array, normalize=False, **options):
        r"""
        Given C in R^{n_clusters \times n}, form A = UC in R^{n_points \times n}

        If `normalize` is ``True``, form A = U(U^TU)^{-1}C to normalize
        the rows of the output (this preserves the 1-norm of each column
        of the array under the transformation)

        """
        orient = np.transpose if options.get('transpose', False) else lambda A: A
        array = orient(array)
        shape = list(array.shape)
        shape[0] = self.n_points
        output = np.zeros(shape)

        for idx_pts, idx_clu in enumerate(self.assignments):
            output[idx_pts, ...] += array[idx_clu, ...]
            if normalize and self.cluster_weights[idx_clu] > 0:
                output[idx_pts, ...] /= self.cluster_weights[idx_clu]

        return orient(output)

def _resample(transform, inputs):
    """ apply upsampling or downsampling to one or more input arrays
    """
    if isinstance(inputs, collections.Mapping):
        outputs = {k: transform(inputs[k]) for k in inputs}
    else:
        try:
            outputs =  type(inputs)(transform(arr) for arr in inputs)
        except:
            outputs = transform(inputs)
    return outputs

def downsample(cluster_map, inputs, normalize=True, **options):
    """ apply `ClusterMap.downsample` to one or more input arrays
    """
    opts = dict(normalize=normalize)
    opts.update(options)

    if isinstance(cluster_map, collections.Mapping):
        output = {k: downsample(cluster_map[k], inputs[k], **opts) for k in cluster_map}
    else:
        transform = functools.partial(cluster_map.downsample, **opts)
        output = _resample(transform, inputs)
    return  output

def upsample(cluster_map, inputs, normalize=False, **options):
    """ apply `ClusterMap.upsample` to one or more input arrays
    """
    opts = dict(normalize=normalize)
    opts.update(options)

    if isinstance(cluster_map, collections.Mapping):
        output = {k: upsample(cluster_map[k], inputs[k], **opts) for k in cluster_map}
    else:
        transform = functools.partial(cluster_map.upsample, **opts)
        output = _resample(transform, inputs)
    return output

def rebase(cluster_map):
    assignments = 1 * cluster_map.vec
    n_clusters = len(set(assignments))

    if n_clusters == cluster_map.n_clusters:
        output = cluster_map
    else:
        reassign = {old: new for new, old in enumerate(sorted(set(assignments)))}
        for i, a in enumerate(assignments):
            assignments[i] = reassign[a]

        output = ClusterMap(cluster_map.n_points, n_clusters, assignments)

    return output

