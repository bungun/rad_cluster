""" TODO: docstring
"""
from __future__ import print_function
import operator
import numpy as np

class DoseConstraint(object):
    """ TODO: docstring
    """
    def __init__(self, relop, dose, statistic, stat):
        if relop in ('<', '<=', operator.lt, operator.le):
            self.relop = operator.le
        elif relop in ('>', '>=', operator.gt, operator.ge):
            self.relop = operator.ge
        else:
            raise ValueError('invalid relop, must be `>` or `<`')
        self.dose = max(0., float(dose))
        self.dose_norm = 1.
        self.statistic = statistic
        self.stat_string = str(stat)

    def satisfied_by(self, voxel_doses, verbose=True, **options):
        """ Test if vector ``voxel_doses`` satisfies the dose constraint
        """
        MINTOL = 0.05 # 0.05 Gy
        percent_tolerance = options.get('percent_tolerance', 1)
        reltol = abs(float(percent_tolerance) / 100.)
        abstol = options.get('dose_tolerance', MINTOL)
        tol = max(reltol * self.dose, abstol)
        tol *= -1 if self.relop is operator.ge else 1

        dose_achieved = self.statistic(voxel_doses) * self.dose_norm
        dose_tolerated = self.dose + tol
        if verbose:
            msg = "constraint: {}\nallowed dose + tolerance: {:0.2f}\nactual dose: {:0.2f}"
            print(msg.format(self, dose_tolerated, dose_achieved))
        return self.relop(dose_achieved, dose_tolerated)

    def __repr__(self):
        relop_str = '>' if self.relop is operator.ge else '<'
        return 'D{} {} {}'.format(self.stat_string, relop_str, self.dose)

    def __str__(self):
        relop_str = '>' if self.relop is operator.ge else '<'
        return 'D("{}") {} {}'.format(self.stat_string, relop_str, self.dose)

class MeanDoseConstraint(DoseConstraint):
    """ TODO: docstring
    """
    def __init__(self, relop, dose):
        DoseConstraint.__init__(self, relop, dose, np.mean, 'mean')

class MinDoseConstraint(DoseConstraint):
    """ TODO: docstring
    """
    def __init__(self, dose):
        DoseConstraint.__init__(self, operator.ge, dose, np.min, 'min')

class MaxDoseConstraint(DoseConstraint):
    """ TODO: docstring
    """
    def __init__(self, dose):
        DoseConstraint.__init__(self, operator.le, dose, np.max, 'max')

class DoseVolumeConstraint(DoseConstraint):
    """ TODO: docstring

        A "dose volume histogram" (DVH) is a graph of the points (d, v)
        such that v% of a structure's volume receives a dose >= d.

        % Volume

        100
           | ..
           |   '.
           |     '..
         v - - - - -x .
           |        |  '..
           |________|_____'____ Dose
          0         d

        (i.e., 100 * (1 - cumulative sum) of a standard histogram of the
        structure's voxel doses.)

        The expression D(x) maps percentiles to doses. For instance,
        D(30%), corresponds to the minimum dose received by the top
        30% most-irradiated portion of the structure.

        Similarly, V(y) maps doses to percentiles. For instance, V(40 Gy)
        corresponds to the volume receiving at least 40 Gy of dose.

        Therefore, the statements

            D(30%) < 40 Gy
            [dose to top 30% of structure receives  is <= 40 Gy]

        and

            V(40 Gy) < 30%
            [volume receiving >= 40 Gy is <= 30% of structure]

        express the same dose volume constraint.
    """
    def __init__(self, percentile, relop, dose):
        self.percentile = percentile = max(0., min(100., float(percentile)))
        def dose_at_percentile(voxel_doses):
            doses = np.hstack((sorted(voxel_doses, reverse=True), [0]))
            percentiles = np.linspace(0, 100, len(doses + 1))#[:-1]
            return np.interp(percentile, percentiles, doses)
        DoseConstraint.__init__(self, relop, dose, dose_at_percentile, percentile)

class DoseConstraintPartial(object):
    """ Build dose-volume constraints of form D(volume_percentile) < dose
    """
    def __init__(self, threshold):
        self.threshold = threshold

    def build_constraint(self, relop, dose):
        """ TODO: docstring
        """
        if self.threshold in ('Min', 'min'):
            return MinDoseConstraint(dose)
        elif self.threshold in ('Max', 'max'):
            return MaxDoseConstraint(dose)
        elif self.threshold in ('Mean', 'mean'):
            return MeanDoseConstraint(relop, dose)
        else:
            return DoseVolumeConstraint(self.threshold, relop, dose)

    def __ge__(self, dose):
        return self.build_constraint(operator.ge, dose)

    def __gt__(self, dose):
        return self.__ge__(dose)

    def __le__(self, dose):
        return self.build_constraint(operator.le, dose)

    def __lt__(self, dose):
        return self.__le__(dose)

class VolumeConstraintPartial(object):
    """ Build dose-volume constraints of form V(dose_level) < volume
    """
    def __init__(self, dose):
        self.dose = dose

    def build_constraint(self, relop, percentile):
        """ TODO: docstring
        """
        return DoseConstraintPartial(percentile).build_constraint(relop, self.dose)

    def __ge__(self, percentile):
        return self.build_constraint(operator.ge, percentile)

    def __gt__(self, percentile):
        return self.__ge__(percentile)

    def __le__(self, percentile):
        return self.build_constraint(operator.le, percentile)

    def __lt__(self, percentile):
        return self.__le__(percentile)

D = DoseConstraintPartial
V = VolumeConstraintPartial

MEAN = 'mean'
MAX = 'max'
MIN = 'min'

def string_to_constraint(string):
    return eval(string)

