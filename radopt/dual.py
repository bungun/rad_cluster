""" TODO: docstring
"""
import operator
import collections
import six
import numpy as np
import optkit as ok

def A_transpose_nu(dose_matrices, nu):
    return sum((dose_matrices[s].T.dot(nu[s]) for s in nu))

def build_objectives(shape, block_sizes, structures, dose_matrices, **options):
    r""" Build dual problem given input dose matrices and structure objectives

        In particular, we build the following problem:

            min  :math:`f_conj(nu)`
            s.t. :math:`A'(\nu) = \mu`
                 :math:`\mu \ge 0`
                 :math:`\nu \in dom(f_conj)`.

        When the keyword argument ``nu_offset`` is provided, we
        build the perturbed problem,

            min  :math:`f^*(\delta + \nu_{offset})`
            s.t. :math:`A'(\delta + \nu_{offset}) = \mu`
                 :math:`\mu \ge 0`
                 :math:`\delta + \nu_{offset} \in dom(f^*)`.

        To rephrase this such that the equality constraint is in
        terms of :math:`\delta` only, introduce a variable
        :math:`u = \mu - A'\nu_{offset}. We then have

            min  :math:`f_conj(\delta; \nu_{offset})`
            s.t. :math:`A'(\delta) = u`
                 :math:`u + \mu_{offset} \ge 0`
                 :math:`\delta + \nu_{offset} \in dom(f^*)`,

        with :math:`mu_offset = A'nu_offset`.
    """
    assert isinstance(block_sizes, collections.Mapping), "block sizes given as dict"
    assert isinstance(structures, collections.Mapping), "structures given as dict"
    assert isinstance(dose_matrices, collections.Mapping), "dose matrices given as dict"
    assert len(set(structures.keys() + dose_matrices.keys())) == len(structures), (
        "all keys shared between structures and dose matrices")
    assert len(set(structures.keys() + dose_matrices.keys())) == len(structures), (
        "all keys shared between structures and dose matrices")

    norm_options = options.pop('norm_options', dict())
    voxel_weights = options.pop('voxel_weights', dict())
    nu_offset = options.pop('nu_offset', dict())
    mu_offset = options.pop('mu_offset', 0)
    assert isinstance(voxel_weights, dict), "voxel weights given as dict"
    assert isinstance(nu_offset, dict), "nu offsets given as dict"

    m, n = shape

    obj_voxels_dual = ok.api.PogsObjective(m)
    obj_beams_dual = ok.api.PogsObjective(n, h='IndGe0', b=-mu_offset)

    dual_evals = dict()
    dual_constraints = dict()
    blk_slices = dict()

    ptr = 0
    for blk_id in structures:
        dual_options = dict()
        dual_options.update(norm_options.get(blk_id, dict()))
        dual_options.update(options)
        if blk_id in voxel_weights:
            dual_options['voxel_weights'] = voxel_weights[blk_id]
        if blk_id in nu_offset:
            dual_options['nonnegative'] = True
            dual_options['nu_offset'] = nu_offset[blk_id]

        size = block_sizes[blk_id]
        s = structures[blk_id]
        blk_slices[blk_id] = slice(ptr, ptr + size)
        obj_sub, eval_sub, constr_sub = s.objective.build_dual(size, **dual_options)
        obj_voxels_dual.copy_from(obj_sub, start_index_target=ptr)
        dual_evals[blk_id] = eval_sub
        dual_constraints[blk_id] = constr_sub
        ptr += size

    def dual_eval(nu):
        if not isinstance(nu, dict):
            nu = {blk_id: nu[blk_slices[blk_id]] for blk_id in structures}
        def eval_block(blk):
            return dual_evals[blk](nu[blk])
        return sum(map(eval_block, nu.keys()))

    def dual_feasible(nu, tol, verbose=True):
        try:
            domain_tol, cone_tol = tol
        except:
            cone_tol = domain_tol = tol

        if not isinstance(nu, dict):
            nu = {blk_id: nu[blk_slices[blk_id]] for blk_id in structures}

        infeas_domain = list()
        for k in structures.keys():
            if not dual_constraints[k](nu[k], domain_tol):
                infeas_domain.append(k)
        feas_domain = len(infeas_domain) == 0
        if not feas_domain and verbose:
            msg = 'Nu infeasible. nu \in dom(f^*) violated for structures {}'
            print(msg.format(infeas_domain))

        mu = A_transpose_nu(dose_matrices, nu)
        feas_cone = all(mu + mu_offset >= -(cone_tol + 1e-8))
        if not feas_cone and verbose:
            msg = 'Mu infeasible. mu_min= {}. Required: mu >= {}'
            print(msg.format(min(mu + mu_offset), -cone_tol))
        return feas_domain, feas_cone

    return obj_voxels_dual, obj_beams_dual, dual_eval, dual_feasible
