import collections
import numpy as np
import optkit as ok

def matvec(A, x):
    if isinstance(A, np.ndarray):
        return np.dot(A, x)
    else:
        return np.squeeze(np.array(A * x))

def A_mul_x(dose_matrices, x, dose_norm=1.0):
    x = float(dose_norm) * x
    return {k: matvec(dose_matrices[k], x) for k in dose_matrices}

def build_objectives(shape, block_sizes, structures, dose_matrices, **options):
    assert isinstance(block_sizes, collections.Mapping), "block sizes given as dict"
    assert isinstance(structures, collections.Mapping), "structures given as dict"
    assert isinstance(dose_matrices, collections.Mapping), "dose matrices given as dict"
    assert len(set(structures.keys()) & set(dose_matrices.keys())) == len(structures), (
        "all keys shared between structures and dose matrices")
    assert len(set(structures.keys()) & set(dose_matrices.keys())) == len(structures), (
        "all keys shared between structures and dose matrices")

    norm_options = options.pop('norm_options', dict())
    voxel_weights = options.pop('voxel_weights', dict())
    assert isinstance(voxel_weights, dict), "voxel weights given as dict"

    m, n = shape

    obj_voxels = ok.api.PogsObjective(m)
    obj_beams = ok.api.PogsObjective(n, h='IndGe0')
    primal_evals = dict()

    ptr = 0
    for blk_id in structures:
        primal_options = dict()
        primal_options.update(norm_options.get(blk_id, dict()))
        primal_options.update(options)
        if blk_id in voxel_weights:
            primal_options['voxel_weights'] = voxel_weights[blk_id]

        size = block_sizes[blk_id]
        s = structures[blk_id]
        obj_sub, eval_sub = s.objective.build_primal(size, **primal_options)
        obj_voxels.copy_from(obj_sub, start_index_target=ptr)
        primal_evals[blk_id] = eval_sub
        ptr += size

    def primal_eval(x):
        def eval_block(blk):
            return primal_evals[blk](np.dot(dose_matrices[blk], x))
        return sum(map(eval_block, structures.keys()))

    return obj_voxels, obj_beams, primal_eval