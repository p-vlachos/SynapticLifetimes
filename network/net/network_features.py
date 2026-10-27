from brian2 import PoissonGroup, Synapses, NeuronGroup, BrianObject
from brian2.units import *
import numpy as np


def set_summed_variable_clock(syn, varname, dt, order=None):
    """
        Brian2 updates summed variables on the clock of the target group (for the synapses onto
        excitatory neurons: every time step, for millions of synapses), while some are only needed
        at longer intervals. Brian2 has no public interface for this, so the (private) clock of
        the updater is replaced. Optionally also set the order within the time step.
    """
    from brian2 import Clock
    updater = syn.summed_updaters.get(varname)
    if updater is None or not hasattr(updater, '_clock'):
        raise RuntimeError(f"Cannot set the clock of the summed variable {varname}; "
                           "the Brian2 internals have changed")
    updater._clock = Clock(dt=dt)
    if order is not None:
        updater.order = order


def distance_expression(grid_wrap):
    """
        Brian2 expression for the distance between the pre- and postsynaptic neuron of a synapse
        (positions x, y), on a torus of side length grid_size if grid_wrap.
    """
    if not grid_wrap:
        return "sqrt((x_pre - x_post)**2 + (y_pre - y_post)**2)"
    def wrapped(c):
        d = f"abs({c}_pre - {c}_post)"
        return f"((grid_size - {d})*int({d} > 0.5*grid_size) + {d}*int({d} <= 0.5*grid_size))"
    return f"sqrt({wrapped('x')}**2 + {wrapped('y')}**2)"
