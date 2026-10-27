#ifndef OUTPUT_FILES_H
#define OUTPUT_FILES_H

#include <ostream>

std::ostream& turnover_output_file();
std::ostream& scaling_deltas_output_file();
std::ostream& scaling_deltas_EI_output_file();
std::ostream& spk_register_output_file();
std::ostream& spk_register_EI_output_file();

// weight snapshots of the active synapses (see cpp_methods.record_weights)
void write_weight_record(int rec_id, double t, int i, int j, double a, int syn_active);

#endif // OUTPUT_FILES_h