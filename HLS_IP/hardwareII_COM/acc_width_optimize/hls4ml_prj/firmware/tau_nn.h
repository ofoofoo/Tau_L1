#ifndef TAU_NN_H_
#define TAU_NN_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"

// Prototype of top level function for C-synthesis
void tau_nn(
    input_t model_input[N_INPUT_1_1],
    result_t layer19_out[N_LAYER_17], result_t layer20_out[N_LAYER_20]
);

#endif
