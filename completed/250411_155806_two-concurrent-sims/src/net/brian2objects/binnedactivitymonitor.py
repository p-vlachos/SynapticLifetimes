import numpy as np
from brian2.devices.cpp_standalone import CPPStandaloneCodeObject

from brian2.units.fundamentalunits import DIMENSIONLESS
from brian2.utils.logger import get_logger
from brian2 import Nameable
from brian2.core.spikesource import SpikeSource
from brian2.core.variables import Variables
from brian2.groups.group import CodeRunner, Group

logger = get_logger(__name__)

CPPStandaloneCodeObject.templater = CPPStandaloneCodeObject.templater.derive(
    "net.brian2objects"
)


class BinnedActivityMonitor(Group, CodeRunner):

    invalidates_magic_network = False
    add_to_magic_network = True

    def __init__(self, source, name='binnedactivitymonitor*', codeobj_class=None, dtype=np.uint64):
        self.source = source

        self.codeobj_class = codeobj_class
        CodeRunner.__init__(self, group=self, code='', template='binnedactivitymonitor',
                            clock=source.clock, when='end', order=0, name=name)

        self.add_dependency(source)

        print(source.N)
        assert source.N <= np.iinfo(dtype).max, "Need to choose dtype that is large enough for all neurons in population active."
        if np.min_scalar_type(source.N) != dtype:
            print(f"Warning: {BinnedActivityMonitor.__name__} could use smaller dtype {np.min_scalar_type(source.N)} instead of {dtype}.")

        self.variables = Variables(self)
        # Handle subgroups correctly
        start = getattr(source, 'start', 0)
        stop = getattr(source, 'stop', len(source))
        self.variables.add_constant('_source_start', start)
        self.variables.add_constant('_source_stop', stop)
        self.variables.add_reference('_spikespace', source)
        self.variables.add_dynamic_array('spk_count', size=0, dimensions=DIMENSIONLESS, read_only=True, dtype=dtype)
        self.variables.add_array('N', dtype=np.int32, size=1, scalar=True, read_only=True)
        self.variables.create_clock_variables(self._clock, prefix='_clock_')
        self._enable_group_attributes()

    def resize(self, new_size):
        # Note that this does not set N, this has to be done in the template
        # since we use a restricted pointer to access it (which promises that
        # we only change the value through this pointer)
        self.variables['spk_count'].resize(new_size)

    def __repr__(self):
        description = '<{classname}, recording {source}>'
        return description.format(classname=self.__class__.__name__,
                                  source=self.source.name)
