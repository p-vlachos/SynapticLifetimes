#ifndef OUTPUT_FILES_H
#define OUTPUT_FILES_H

#include <ostream>

std::ostream& turnover_output_file();
std::ostream& turnover_EI_output_file();
std::ostream& scaling_deltas_output_file();
std::ostream& scaling_deltas_EI_output_file();
std::ostream& spk_register_output_file();
std::ostream& spk_register_EI_output_file();

#endif // OUTPUT_FILES_h