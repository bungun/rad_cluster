import operator
import radopt as ro

def test_string_to_constraints():
    constr = ro.constraint.string_to_constraint('D(MEAN) > 50')
    assert isinstance(constr, ro.constraint.DoseConstraint)
    assert isinstance(constr, ro.constraint.MeanDoseConstraint)
    assert constr.dose == 50
    assert constr.relop is operator.ge

    constr = ro.constraint.string_to_constraint('D("max") < 20')
    assert isinstance(constr, ro.constraint.DoseConstraint)
    assert isinstance(constr, ro.constraint.MaxDoseConstraint)
    assert constr.dose == 20
    print constr.relop
    assert constr.relop is operator.le

    c = ro.constraint.D(25) > 30
    constr = ro.constraint.string_to_constraint(str(c))
    assert isinstance(constr, ro.constraint.DoseConstraint)
    assert isinstance(constr, ro.constraint.DoseVolumeConstraint)
    assert constr.dose == 30
    assert constr.percentile == 25
    assert constr.relop is operator.ge
