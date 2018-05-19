import radopt as ro

def test_structure():
    targ = ro.structure.Target('target', 100)
    assert isinstance(targ.objective, ro.objective.ObjectivePWL)
    assert targ.name == 'target'
    assert targ.size == 100
    assert targ.objective.parameters['dose'] == 1

    targ = ro.structure.Target(
            'target', 100,
            dose=60, weight=5,
            weight_underdose=10,
            weight_overdose=2)
    assert isinstance(targ.objective, ro.objective.ObjectivePWL)
    assert targ.objective.parameters['dose'] == 60
    assert 'weight' not in targ.objective.parameters
    assert targ.objective.parameters['weight_underdose'] == 10
    assert targ.objective.parameters['weight_overdose'] == 2

    oar = ro.structure.OAR('oar', 100, weight=2.3)
    assert oar.name == 'oar'
    assert oar.size == 100
    assert isinstance(oar.objective, ro.objective.ObjectiveLinear)
    assert oar.objective.parameters.get('weight') == 2.3

    targ2 = ro.structure.dict_to_structure(**targ.dict())
    assert targ2.name == targ.name
    assert targ2.size == targ.size
    assert targ2.objective.parameters == targ.objective.parameters

    oar2 = ro.structure.dict_to_structure(**oar.dict())
    assert oar2.name == oar.name
    assert oar2.size == oar.size
    assert oar2.objective.parameters == oar.objective.parameters

    targ3 = ro.structure.dict_to_structure(**eval(str(targ.dict())))
    assert targ3.objective.parameters == targ.objective.parameters
