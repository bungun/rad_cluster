import numpy as np
import radopt as ro

def test_problem():
    TOL = 1e-10
    voxels, beams = 110, 50
    ovox = 10
    tvox = voxels - ovox
    tvox1 = int(0.6 * tvox)
    tvox2 = tvox - tvox1
    structures = {
            0: ro.structure.Target('target', tvox1),
            1: ro.structure.Target('target2', tvox2),
            2: ro.structure.OAR('oar', ovox)}
    structures[0].objective.parameters['weight_underdose'] = np.random.uniform(1, 2)
    structures[0].objective.parameters['dose'] = np.random.uniform(1, 2)
    dose_matrices = {k: np.random.random((structures[k].size, beams)) for k in structures}
    prb = ro.problem.Problem()
    prb.structures.update(structures)

    # test name_to_id
    for k in structures:
        assert prb.name_to_id(k) == k
        assert prb.name_to_id(structures[k].name) == k

    # test dim calcs
    assert prb.calc_row_dim(dose_matrices)[0] == tvox + 1
    assert prb.calc_col_dim(dose_matrices) == beams

    # test key order
    prb.set_key_order(key_order=[2, 1, 0])
    assert prb.structures.keys() == [2, 1, 0]
    prb.set_key_order(key_order=['target', 'target2', 'oar'])
    assert prb.structures.keys() == [0, 1, 2]

    # test joint normalization
    # default behavior: for each structure S, normalize weights by |S|/|S_key|
    #  for S = S_key, the normalization is 1
    #  for compressed structures, replace |S| by 1
    norm_opts = prb.joint_normalization('target')
    assert all((k in norm_opts for k in structures))
    assert abs(norm_opts[0]['dose_norm'] - 1) < TOL
    assert abs(norm_opts[1]['dose_norm'] - 1) < TOL
    assert abs(norm_opts[2]['dose_norm'] - 1) < TOL
    assert abs(norm_opts[0]['weight_norm'] - tvox1) < TOL
    assert abs(norm_opts[1]['weight_norm'] - tvox2) < TOL
    assert abs(norm_opts[2]['weight_norm'] - 1.) < TOL

    # optional behavior: for each structure S, normalize weights by weight_key * |S|/|S_key|
    #  for S = S_key, the normalized weights = 1
    wt = structures[0].objective.parameters['weight_underdose']
    norm_opts = prb.joint_normalization('target', key_wt='weight_underdose')
    assert abs(norm_opts[0]['dose_norm'] - 1) < TOL
    assert abs(norm_opts[1]['dose_norm'] - 1) < TOL
    assert abs(norm_opts[2]['dose_norm'] - 1) < TOL
    assert abs(norm_opts[0]['weight_norm'] - wt*tvox1) < TOL
    assert abs(norm_opts[1]['weight_norm'] - wt*tvox2) < TOL
    assert abs(norm_opts[2]['weight_norm'] - wt) < TOL

    # optional behavior: for each structure S, normalize doses by dose_key
    #  for S = S_key, the normalized dose = 1
    dz = structures[0].objective.parameters['dose']
    norm_opts = prb.joint_normalization('target', key_wt='weight_underdose', key_dose='dose')
    assert abs(norm_opts[0]['dose_norm'] - dz) < TOL
    assert abs(norm_opts[1]['dose_norm'] - dz) < TOL
    assert abs(norm_opts[2]['dose_norm'] - dz) < TOL
    assert abs(norm_opts[0]['weight_norm'] - wt*tvox1) < TOL
    assert abs(norm_opts[1]['weight_norm'] - wt*tvox2) < TOL
    assert abs(norm_opts[2]['weight_norm'] - wt) < TOL

    # normalize problem
    # default behavior: normalize dose, not weights
    prb_opts = prb.normalize_problem()
    norm_opts = prb_opts.get('norm_options')
    wt = structures[0].objective.parameters['weight_underdose']
    dz = structures[0].objective.parameters['dose']
    assert abs(norm_opts[0]['dose_norm'] - dz) < TOL
    assert abs(norm_opts[1]['dose_norm'] - dz) < TOL
    assert abs(norm_opts[2]['dose_norm'] - dz) < TOL
    assert abs(norm_opts[0]['weight_norm'] - tvox1) < TOL
    assert abs(norm_opts[1]['weight_norm'] - tvox2) < TOL
    assert abs(norm_opts[2]['weight_norm'] - 1) < TOL

    prb_opts = prb.normalize_problem(key_structure=1)
    norm_opts = prb_opts.get('norm_options')
    wt = structures[1].objective.parameters['weight_underdose']
    dz = structures[1].objective.parameters['dose']
    assert abs(norm_opts[0]['dose_norm'] - dz) < TOL
    assert abs(norm_opts[1]['dose_norm'] - dz) < TOL
    assert abs(norm_opts[2]['dose_norm'] - dz) < TOL
    assert abs(norm_opts[0]['weight_norm'] - tvox1) < TOL
    assert abs(norm_opts[1]['weight_norm'] - tvox2) < TOL
    assert abs(norm_opts[2]['weight_norm'] - 1) < TOL

    # contiguous <-> blocks
    rowdim, block_sizes = prb.calc_row_dim(dose_matrices)
    y_vec = np.random.random(rowdim)
    y = prb.contig_to_blocks(y_vec, block_sizes)
    assert y.keys() == prb.structures.keys()
    assert all(y[k].shape[0] == block_sizes[k] for k in structures)
    y_vec_prime = prb.blocks_to_contig(y)
    assert np.sum(y_vec - y_vec_prime) == 0

    # build matrix
    A, wrk = prb.build_matrix(dose_matrices)
    assert A.shape == (rowdim, beams)
    assert wrk.keys() == dose_matrices.keys()

    # build primal objectives
    ff, gg, _ = prb.build_primal_objectives(dose_matrices)
    assert ff.size == tvox + 1
    assert gg.size == beams

    # build dual objectives
    ff_star, gg_star, _, _= prb.build_dual_objectives(dose_matrices)
    assert ff_star.size == tvox + 1
    assert gg_star.size == beams

