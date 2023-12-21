#include "top.h"

void wrapper(ap_uint<CFG_AW * CFG_TOP_PIC> *act_in,
             ap_int<32> *act_out, ap_int<CFG_MW> *mask,
             int num_nz) {
#pragma HLS DATAFLOW
    /*gen_code-fifo*/

    /*gen_code-load*/

    /*gen_code-comp*/

    /*gen_code-store*/
}

void top(ap_uint<CFG_AW * CFG_TOP_PIC> *act_in, ap_int<32> *act_out,
         ap_int<CFG_MW> *mask, int num_nz) {
#pragma HLS INTERFACE m_axi port = act_in offset = slave bundle = \
    gmem0 depth = 65536
#pragma HLS INTERFACE m_axi port = act_out offset = slave bundle = \
    gmem1 depth = 65536
#pragma HLS INTERFACE m_axi port = mask offset = slave bundle = gmem2 depth = \
    65536
#pragma HLS INTERFACE s_axilite port = act_in bundle = control
#pragma HLS INTERFACE s_axilite port = act_out bundle = control
#pragma HLS INTERFACE s_axilite port = mask bundle = control
#pragma HLS INTERFACE s_axilite port = num_nz bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    wrapper(act_in, act_out, mask, num_nz);
}
