#include "output_files.h"

#include <fstream>

namespace {
    /**
     * These objects live in a translation unit, i.e. .o file.
     * The functions of this file are exposed to other .o files, i.e.
     * the different functions in cpp_methods.py which are emitted to multiple
     * translation units (because they can are used in multiple Brian2 groups).
     * Now if each translation unit had their own version of these objects, there
     * would be data races (see 70fb7c1c5c89c5e256c1db0c7b9ed384c3f1a7b7).
     * This way the objects live in only one translation unit, the one for this cpp file
     * and all other units link to this unit so they all share the same ofstream.
     * All of this of course only works because there is no parallelization within the
     * simulation.
     **/
    std::ofstream          turnover_output_stream{"turnover",          std::ios_base::app};
    std::ofstream       turnover_EI_output_stream("turnover_EI",       std::ios_base::app);
    std::ofstream    scaling_deltas_output_stream("scaling_deltas",    std::ios_base::app);
    std::ofstream scaling_deltas_EI_output_stream{"scaling_deltas_EI", std::ios_base::app};
    std::ofstream      spk_register_output_stream{"spk_register",      std::ios_base::app};
    std::ofstream   spk_register_EI_output_stream("spk_register_EI",   std::ios_base::app);
}

std::ostream& turnover_output_file() {
    return turnover_output_stream;
}

std::ostream& turnover_EI_output_file() {
    return turnover_EI_output_stream;
}

std::ostream& scaling_deltas_output_file() {
    return scaling_deltas_output_stream;
}

std::ostream& scaling_deltas_EI_output_file() {
    return scaling_deltas_EI_output_stream;
}

std::ostream& spk_register_output_file() {
    return spk_register_output_stream;
}

std::ostream& spk_register_EI_output_file() {
    return spk_register_EI_output_stream;
}