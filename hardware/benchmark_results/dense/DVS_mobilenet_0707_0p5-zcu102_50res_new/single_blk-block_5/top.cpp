#include "top.h"

void wrapper(ap_int<CFG_AW * CFG_TOP_PIC> *act_in,
             ap_int<CFG_AW * CFG_TOP_POC> *act_out, ap_int<CFG_MW> *mask,
             int num_nz) {
#pragma HLS DATAFLOW
    /*gen_code-fifo*/
    hls::stream<BundleT<CFG_TOP_PIC, ap_int<CFG_AW>>> a_top;
#pragma HLS STREAM variable=a_top depth=2
    hls::stream<BundleT<CFG_BLOCK_5_POC, ap_int<CFG_AW>>> a_block_5;
#pragma HLS STREAM variable=a_block_5 depth=2

    /*gen_code-load*/
    read_sparse_input<CFG_TOP_PIC, CFG_AW, CFG_TOP_IC, CFG_TOP_IH, CFG_TOP_IW>(act_in, a_top);

    /*gen_code-comp*/
    conv_1x1_3x3_dw_1x1_stride1_residual<CFG_BLOCK_5_PIC, CFG_BLOCK_5_PC, CFG_BLOCK_5_POC, CFG_BLOCK_5_IC, CFG_BLOCK_5_C, CFG_BLOCK_5_H, CFG_BLOCK_5_W, CFG_BLOCK_5_PW0, CFG_BLOCK_5_PW1, CFG_BLOCK_5_PW2, CFG_BLOCK_5_SW0, CFG_BLOCK_5_SW1, CFG_BLOCK_5_SW2, CFG_BLOCK_5_BW0, CFG_BLOCK_5_BW1, CFG_BLOCK_5_BW2, CFG_BLOCK_5_IW, CFG_AW, CFG_WW, CFG_EXP>(a_top, a_block_5, w_block_5_0, w_block_5_1, w_block_5_2, s_block_5_0, s_block_5_1, s_block_5_2, i_block_5);

    /*gen_code-store*/
    write_output<CFG_TOP_POC, CFG_TOP_OC, CFG_AW, CFG_TOP_OH, CFG_TOP_OW>(a_block_5, act_out);
}

void top(ap_int<CFG_AW * CFG_TOP_PIC> *act_in, ap_int<CFG_AW * CFG_TOP_POC> *act_out,
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
