import numpy as np

def dvh(voxel_doses):
    doses = np.hstack(([0], np.sort(voxel_doses)))
    percentiles = np.linspace(100, 0, len(doses + 1))#
    return (doses, percentiles)