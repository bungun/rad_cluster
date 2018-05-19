""" TODO: docstring
"""
import six
import abc

from radopt import objective

@six.add_metaclass(abc.ABCMeta)
class Structure(object):
    """ Component of a treatment planning problem
    """
    def __init__(self, name, size, obj, **kwargs):
        self.name = str(name)
        self.size = int(size)
        self.objective = objective.OBJECTIVE_CONSTRUCTORS[obj](**kwargs)

    @abc.abstractmethod
    def collapsable(self, **options):
        raise NotImplementedError

    def __repr__(self):
        return '{} ({}), objective: {}'.format(
                self.name, type(self).__name__, repr(self.objective))

    def __str__(self):
        return '{}:\n\ttype: {}\n\tsize: {}\n\tobjective: {}'.format(
                self.name, type(self).__name__, self.size, self.objective)

    def dict(self):
        return dict(
                name=self.name,
                type=type(self).__name__,
                size=self.size,
                objective=objective.OBJECTIVE_NAMES[repr(self.objective)],
                parameters=self.objective.parameters)

class Target(Structure):
    """ Subclass structure to treatment targets
    """
    def __init__(self, name, size, **kwargs):
        obj = kwargs.pop('objective', 'piecewise_linear')
        Structure.__init__(self, name, size, obj, **kwargs)

    def collapsable(self, **options):
        """ Target structures are never collapsable blocks of a problem
        """
        return False

class OAR(Structure):
    """ Subclass structure to treatment organs-at-risk
    """
    def __init__(self, name, size, **kwargs):
        obj = kwargs.pop('objective', 'linear')
        Structure.__init__(self, name, size, obj, **kwargs)

    def collapsable(self, **options):
        """ OAR structures collapsable if unconstrained & have linear objective
        """
        return (
                isinstance(self.objective, objective.ObjectiveLinear)
                and options.get('collapse', True))

def dict_to_structure(**kwargs):
    name = kwargs.pop('name')
    constructor = eval(kwargs.pop('type'))
    size = kwargs.pop('size')
    params = kwargs.pop('parameters')
    return constructor(name, size, **params)

def from_dict(dictionary):
    return dict_to_structure(**dictionary)

def from_string(string):
    return from_dict(eval(string))