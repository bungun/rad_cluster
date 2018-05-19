""" TODO: docstring
"""
from __future__ import print_function
import functools
import numpy as np
import radopt.primal

def suboptimality(result):
    upper_bound = result['upper_bound']
    lower_bound = result['lower_bound']
    return 100 * (upper_bound - lower_bound) / upper_bound

class ClinicallyFeasiblePareto(object):
    """ TODO: docstring
    """
    def __init__(self, **options):
        self.constraints = dict()
        self.constraints.update(options.pop('dose_constraints', dict()))

    def clinically_feasible(self, constraints, dose_matrices, beam_weights,
                            **options):
        """ TODO: docstring
        """
        verbose = options.pop('verbose', True)
        doses = radopt.primal.A_mul_x(dose_matrices, beam_weights)
        feasible = True
        for k in constraints:
            if verbose:
                msg = "Constraints for structure {}".format(k)
                msg = "\n" + msg + "\n" + len(msg) * "-"
                print(msg)
            for c in constraints[k]:
                feasible &= c.satisfied_by(doses[k], verbose=verbose, **options)
        return feasible

    def explore_weights(self, problem, initial_weights, weight_increments, **options):
        """ TODO: docstring
        """
        solutions = dict()
        write_and_burn = options.get('write_and_burn', lambda key, result: result)
        clinical_matrices = options.pop('clinical_matrices', problem.A)
        verbose_pareto = options.pop('verbose_pareto', True)
        verbose_weights = options.pop('verbose_weights', False)
        verbose_constraints = options.pop('verbose_constraints', True)
        dose_tolerance=options.pop('dose_tolerance', 0.05)
        constraints = options.pop('dose_constraints', dict())
        constraints.update(self.constraints)
        weight_limits = options.pop('weight_limits', dict())
        for key in problem.structures:
            if key not in constraints:
                constraints[key] = tuple()
        clinically_feasible = functools.partial(
                self.clinically_feasible, constraints, clinical_matrices,
                dose_tolerance=dose_tolerance, verbose=verbose_constraints)

        with problem.build_solver(**options) as solver:
            for (structure, parameter) in initial_weights:
                w0 = initial_weights[structure, parameter]
                problem.structures[structure].objective.parameters[parameter] = w0

            result = problem.solve_and_bound(solver, **options)
            if not clinically_feasible(result['beam_weights']):
                raise ValueError('Initial weights infeasible!')

            solutions['nominal'] = write_and_burn('nominal', result)
            options['resume'] = True

            for (structure, parameter) in weight_increments:
                w0 = initial_weights[structure, parameter]
                incr = weight_increments[structure, parameter]
                min_w = w0
                max_w = w0
                try:
                    WMIN, WMAX = weight_limits.get((structure, parameter))
                except:
                    WMIN, WMAX = (1e-8, 1e8)
                if verbose_pareto:
                    msg = (
                            '\nsearching plans for structure={}, parameter={}'
                            ' in interval [{}, {}]')
                    print(msg.format(structure, parameter, WMIN, WMAX))

                runs = 0
                for direction in ('up', 'down'):
                    w = 1. * w0
                    increment = incr if direction == 'up' else 1./incr
                    if verbose_weights:
                        print('increment=', increment)
                    feasible = True
                    while (feasible):
                        # adjust weights
                        w = w * increment
                        if (w > WMAX or w < WMIN):
                            break
                        runs += 1
                        if verbose_weights:
                            print('run=', runs)
                            print('WMIN, w, WMAX: {} <= {} <= {}'.format(WMIN, w, WMAX))
                        min_w = min(w, min_w)
                        max_w = max(w, max_w)
                        w_key = str.format('{:0.2e}', w)
                        problem.structures[structure].objective.parameters[parameter] = w

                        # solve
                        result = problem.solve_and_bound(solver, **options)


                        # assess & store output
                        ptol = min(5, suboptimality(result))
                        feasible = clinically_feasible(
                                result['beam_weights'],
                                percent_tolerance=ptol)
                        if feasible:
                            key = (structure, parameter, w_key)
                            solutions[key] = write_and_burn(key, result)
                    problem.structures[structure].objective.parameters[parameter] = w0

                if verbose_pareto:
                    msg = '\n{} plans away from nominal for structure={}, parameter={}'
                    print(msg.format(runs, structure, parameter))
                    print('weight interval: [{:0.1e}, {:0.1e}]'.format(min_w, max_w))

        return solutions