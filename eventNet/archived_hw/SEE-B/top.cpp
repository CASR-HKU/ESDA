#include "top.h"

void wrapper(ap_int<CFG_AW * CFG_TOP_PIC> *act_in,
             ap_int<32> *act_out, ap_int<CFG_MW> *mask,
             int num_nz) {
#pragma HLS DATAFLOW
    /*gen_code-fifo*/
    hls::stream<BundleT<CFG_TOP_PIC, ap_int<CFG_AW>>> a_top;
#pragma HLS STREAM variable=a_top depth=2
    hls::stream<T_K> t_top;
#pragma HLS STREAM variable=t_top depth=2
    hls::stream<ap_int<CFG_MW>> m_top;
#pragma HLS STREAM variable=m_top depth=20
    hls::stream<T_K> ts2_top;
#pragma HLS STREAM variable=ts2_top depth=128
    hls::stream<BundleT<CFG_CONV1_POC, ap_int<CFG_AW>>> a_conv1;
#pragma HLS STREAM variable=a_conv1 depth=2
    hls::stream<T_K> t_conv1;
#pragma HLS STREAM variable=t_conv1 depth=2
    hls::stream<BundleT<CFG_BLOCK_0_POC, ap_int<CFG_AW>>> a_block_0;
#pragma HLS STREAM variable=a_block_0 depth=2
    hls::stream<T_K> t_block_0;
#pragma HLS STREAM variable=t_block_0 depth=2
    hls::stream<BundleT<CFG_BLOCK_1_POC, ap_int<CFG_AW>>> a_block_1;
#pragma HLS STREAM variable=a_block_1 depth=2
    hls::stream<T_K> t_block_1;
#pragma HLS STREAM variable=t_block_1 depth=2
    hls::stream<BundleT<CFG_BLOCK_2_POC, ap_int<CFG_AW>>> a_block_2;
#pragma HLS STREAM variable=a_block_2 depth=2
    hls::stream<T_K> t_block_2;
#pragma HLS STREAM variable=t_block_2 depth=2
    hls::stream<BundleT<CFG_BLOCK_3_POC, ap_int<CFG_AW>>> a_block_3;
#pragma HLS STREAM variable=a_block_3 depth=2
    hls::stream<T_K> t_block_3;
#pragma HLS STREAM variable=t_block_3 depth=2
    hls::stream<BundleT<CFG_BLOCK_4_POC, ap_int<CFG_AW>>> a_block_4;
#pragma HLS STREAM variable=a_block_4 depth=2
    hls::stream<T_K> t_block_4;
#pragma HLS STREAM variable=t_block_4 depth=2
    hls::stream<BundleT<CFG_BLOCK_5_POC, ap_int<CFG_AW>>> a_block_5;
#pragma HLS STREAM variable=a_block_5 depth=2
    hls::stream<T_K> t_block_5;
#pragma HLS STREAM variable=t_block_5 depth=2
    hls::stream<BundleT<CFG_BLOCK_6_POC, ap_int<CFG_AW>>> a_block_6;
#pragma HLS STREAM variable=a_block_6 depth=2
    hls::stream<T_K> t_block_6;
#pragma HLS STREAM variable=t_block_6 depth=2
    hls::stream<BundleT<CFG_BLOCK_7_POC, ap_int<CFG_AW>>> a_block_7;
#pragma HLS STREAM variable=a_block_7 depth=2
    hls::stream<T_K> t_block_7;
#pragma HLS STREAM variable=t_block_7 depth=2
    hls::stream<BundleT<CFG_BLOCK_8_POC, ap_int<CFG_AW>>> a_block_8;
#pragma HLS STREAM variable=a_block_8 depth=2
    hls::stream<T_K> t_block_8;
#pragma HLS STREAM variable=t_block_8 depth=2
    hls::stream<BundleT<CFG_BLOCK_9_POC, ap_int<CFG_AW>>> a_block_9;
#pragma HLS STREAM variable=a_block_9 depth=2
    hls::stream<T_K> t_block_9;
#pragma HLS STREAM variable=t_block_9 depth=2
    hls::stream<BundleT<CFG_BLOCK_10_POC, ap_int<CFG_AW>>> a_block_10;
#pragma HLS STREAM variable=a_block_10 depth=2
    hls::stream<T_K> t_block_10;
#pragma HLS STREAM variable=t_block_10 depth=2
    hls::stream<BundleT<CFG_BLOCK_11_POC, ap_int<CFG_AW>>> a_block_11;
#pragma HLS STREAM variable=a_block_11 depth=2
    hls::stream<T_K> t_block_11;
#pragma HLS STREAM variable=t_block_11 depth=2
    hls::stream<BundleT<CFG_BLOCK_12_POC, ap_int<CFG_AW>>> a_block_12;
#pragma HLS STREAM variable=a_block_12 depth=2
    hls::stream<T_K> t_block_12;
#pragma HLS STREAM variable=t_block_12 depth=2
    hls::stream<BundleT<CFG_BLOCK_13_POC, ap_int<CFG_AW>>> a_block_13;
#pragma HLS STREAM variable=a_block_13 depth=2
    hls::stream<T_K> t_block_13;
#pragma HLS STREAM variable=t_block_13 depth=2
    hls::stream<BundleT<CFG_CONV8_POC, ap_int<CFG_AW>>> a_conv8;
#pragma HLS STREAM variable=a_conv8 depth=2
    hls::stream<T_K> t_conv8;
#pragma HLS STREAM variable=t_conv8 depth=2
    hls::stream<BundleT<CFG_FC_POC, ap_int<CFG_AW>>> a_fc;
#pragma HLS STREAM variable=a_fc depth=2
    hls::stream<T_K> t_fc;
#pragma HLS STREAM variable=t_fc depth=2

    /*gen_code-load*/
    read_sparse_input<CFG_TOP_PIC, CFG_AW, CFG_TOP_IC>(act_in, a_top, num_nz);
    M2S_mask<CFG_MW, CFG_TOP_IH, CFG_TOP_IW>(mask, t_top, m_top);
    mask_stride2<CFG_MW, CFG_TOP_IH, CFG_TOP_IW>(m_top, ts2_top);

    /*gen_code-comp*/
    conv_3x3_first_layer<CFG_CONV1_PIC, CFG_CONV1_POC, CFG_CONV1_OC, CFG_CONV1_H, CFG_CONV1_W, CFG_CONV1_PW, CFG_CONV1_SW, CFG_CONV1_BW, CFG_AW, CFG_WW, CFG_EXP>(a_top, a_conv1, t_top, t_conv1, ts2_top, w_conv1, s_conv1);
    conv_1x1_3x3_dw_1x1_stride2<CFG_BLOCK_0_PIC, CFG_BLOCK_0_PC, CFG_BLOCK_0_POC, CFG_BLOCK_0_IC, CFG_BLOCK_0_C, CFG_BLOCK_0_OC, CFG_BLOCK_0_H, CFG_BLOCK_0_W, CFG_BLOCK_0_PW0, CFG_BLOCK_0_PW1, CFG_BLOCK_0_PW2, CFG_BLOCK_0_SW0, CFG_BLOCK_0_SW1, CFG_BLOCK_0_SW2, CFG_BLOCK_0_BW0, CFG_BLOCK_0_BW1, CFG_BLOCK_0_BW2, CFG_AW, CFG_WW, CFG_EXP>(a_conv1, a_block_0, t_conv1, t_block_0, w_block_0_0, w_block_0_1, w_block_0_2, s_block_0_0, s_block_0_1, s_block_0_2);
    conv_1x1_3x3_dw_1x1_stride1_residual<CFG_BLOCK_1_PIC, CFG_BLOCK_1_PC, CFG_BLOCK_1_POC, CFG_BLOCK_1_IC, CFG_BLOCK_1_C, CFG_BLOCK_1_H, CFG_BLOCK_1_W, CFG_BLOCK_1_PW0, CFG_BLOCK_1_PW1, CFG_BLOCK_1_PW2, CFG_BLOCK_1_SW0, CFG_BLOCK_1_SW1, CFG_BLOCK_1_SW2, CFG_BLOCK_1_BW0, CFG_BLOCK_1_BW1, CFG_BLOCK_1_BW2, CFG_BLOCK_1_IW, CFG_AW, CFG_WW, CFG_EXP>(a_block_0, a_block_1, t_block_0, t_block_1, w_block_1_0, w_block_1_1, w_block_1_2, s_block_1_0, s_block_1_1, s_block_1_2, i_block_1);
    conv_1x1_3x3_dw_1x1_stride1_residual<CFG_BLOCK_2_PIC, CFG_BLOCK_2_PC, CFG_BLOCK_2_POC, CFG_BLOCK_2_IC, CFG_BLOCK_2_C, CFG_BLOCK_2_H, CFG_BLOCK_2_W, CFG_BLOCK_2_PW0, CFG_BLOCK_2_PW1, CFG_BLOCK_2_PW2, CFG_BLOCK_2_SW0, CFG_BLOCK_2_SW1, CFG_BLOCK_2_SW2, CFG_BLOCK_2_BW0, CFG_BLOCK_2_BW1, CFG_BLOCK_2_BW2, CFG_BLOCK_2_IW, CFG_AW, CFG_WW, CFG_EXP>(a_block_1, a_block_2, t_block_1, t_block_2, w_block_2_0, w_block_2_1, w_block_2_2, s_block_2_0, s_block_2_1, s_block_2_2, i_block_2);
    conv_1x1_3x3_dw_1x1_stride1_residual<CFG_BLOCK_3_PIC, CFG_BLOCK_3_PC, CFG_BLOCK_3_POC, CFG_BLOCK_3_IC, CFG_BLOCK_3_C, CFG_BLOCK_3_H, CFG_BLOCK_3_W, CFG_BLOCK_3_PW0, CFG_BLOCK_3_PW1, CFG_BLOCK_3_PW2, CFG_BLOCK_3_SW0, CFG_BLOCK_3_SW1, CFG_BLOCK_3_SW2, CFG_BLOCK_3_BW0, CFG_BLOCK_3_BW1, CFG_BLOCK_3_BW2, CFG_BLOCK_3_IW, CFG_AW, CFG_WW, CFG_EXP>(a_block_2, a_block_3, t_block_2, t_block_3, w_block_3_0, w_block_3_1, w_block_3_2, s_block_3_0, s_block_3_1, s_block_3_2, i_block_3);
    conv_1x1_3x3_dw_1x1_stride1<CFG_BLOCK_4_PIC, CFG_BLOCK_4_PC, CFG_BLOCK_4_POC, CFG_BLOCK_4_IC, CFG_BLOCK_4_C, CFG_BLOCK_4_OC, CFG_BLOCK_4_H, CFG_BLOCK_4_W, CFG_BLOCK_4_PW0, CFG_BLOCK_4_PW1, CFG_BLOCK_4_PW2, CFG_BLOCK_4_SW0, CFG_BLOCK_4_SW1, CFG_BLOCK_4_SW2, CFG_BLOCK_4_BW0, CFG_BLOCK_4_BW1, CFG_BLOCK_4_BW2, CFG_AW, CFG_WW, CFG_EXP>(a_block_3, a_block_4, t_block_3, t_block_4, w_block_4_0, w_block_4_1, w_block_4_2, s_block_4_0, s_block_4_1, s_block_4_2);
    conv_1x1_3x3_dw_1x1_stride1_residual<CFG_BLOCK_5_PIC, CFG_BLOCK_5_PC, CFG_BLOCK_5_POC, CFG_BLOCK_5_IC, CFG_BLOCK_5_C, CFG_BLOCK_5_H, CFG_BLOCK_5_W, CFG_BLOCK_5_PW0, CFG_BLOCK_5_PW1, CFG_BLOCK_5_PW2, CFG_BLOCK_5_SW0, CFG_BLOCK_5_SW1, CFG_BLOCK_5_SW2, CFG_BLOCK_5_BW0, CFG_BLOCK_5_BW1, CFG_BLOCK_5_BW2, CFG_BLOCK_5_IW, CFG_AW, CFG_WW, CFG_EXP>(a_block_4, a_block_5, t_block_4, t_block_5, w_block_5_0, w_block_5_1, w_block_5_2, s_block_5_0, s_block_5_1, s_block_5_2, i_block_5);
    conv_1x1_3x3_dw_1x1_stride2<CFG_BLOCK_6_PIC, CFG_BLOCK_6_PC, CFG_BLOCK_6_POC, CFG_BLOCK_6_IC, CFG_BLOCK_6_C, CFG_BLOCK_6_OC, CFG_BLOCK_6_H, CFG_BLOCK_6_W, CFG_BLOCK_6_PW0, CFG_BLOCK_6_PW1, CFG_BLOCK_6_PW2, CFG_BLOCK_6_SW0, CFG_BLOCK_6_SW1, CFG_BLOCK_6_SW2, CFG_BLOCK_6_BW0, CFG_BLOCK_6_BW1, CFG_BLOCK_6_BW2, CFG_AW, CFG_WW, CFG_EXP>(a_block_5, a_block_6, t_block_5, t_block_6, w_block_6_0, w_block_6_1, w_block_6_2, s_block_6_0, s_block_6_1, s_block_6_2);
    conv_1x1_3x3_dw_1x1_stride1_residual<CFG_BLOCK_7_PIC, CFG_BLOCK_7_PC, CFG_BLOCK_7_POC, CFG_BLOCK_7_IC, CFG_BLOCK_7_C, CFG_BLOCK_7_H, CFG_BLOCK_7_W, CFG_BLOCK_7_PW0, CFG_BLOCK_7_PW1, CFG_BLOCK_7_PW2, CFG_BLOCK_7_SW0, CFG_BLOCK_7_SW1, CFG_BLOCK_7_SW2, CFG_BLOCK_7_BW0, CFG_BLOCK_7_BW1, CFG_BLOCK_7_BW2, CFG_BLOCK_7_IW, CFG_AW, CFG_WW, CFG_EXP>(a_block_6, a_block_7, t_block_6, t_block_7, w_block_7_0, w_block_7_1, w_block_7_2, s_block_7_0, s_block_7_1, s_block_7_2, i_block_7);
    conv_1x1_3x3_dw_1x1_stride1_residual<CFG_BLOCK_8_PIC, CFG_BLOCK_8_PC, CFG_BLOCK_8_POC, CFG_BLOCK_8_IC, CFG_BLOCK_8_C, CFG_BLOCK_8_H, CFG_BLOCK_8_W, CFG_BLOCK_8_PW0, CFG_BLOCK_8_PW1, CFG_BLOCK_8_PW2, CFG_BLOCK_8_SW0, CFG_BLOCK_8_SW1, CFG_BLOCK_8_SW2, CFG_BLOCK_8_BW0, CFG_BLOCK_8_BW1, CFG_BLOCK_8_BW2, CFG_BLOCK_8_IW, CFG_AW, CFG_WW, CFG_EXP>(a_block_7, a_block_8, t_block_7, t_block_8, w_block_8_0, w_block_8_1, w_block_8_2, s_block_8_0, s_block_8_1, s_block_8_2, i_block_8);
    conv_1x1_3x3_dw_1x1_stride2<CFG_BLOCK_9_PIC, CFG_BLOCK_9_PC, CFG_BLOCK_9_POC, CFG_BLOCK_9_IC, CFG_BLOCK_9_C, CFG_BLOCK_9_OC, CFG_BLOCK_9_H, CFG_BLOCK_9_W, CFG_BLOCK_9_PW0, CFG_BLOCK_9_PW1, CFG_BLOCK_9_PW2, CFG_BLOCK_9_SW0, CFG_BLOCK_9_SW1, CFG_BLOCK_9_SW2, CFG_BLOCK_9_BW0, CFG_BLOCK_9_BW1, CFG_BLOCK_9_BW2, CFG_AW, CFG_WW, CFG_EXP>(a_block_8, a_block_9, t_block_8, t_block_9, w_block_9_0, w_block_9_1, w_block_9_2, s_block_9_0, s_block_9_1, s_block_9_2);
    conv_1x1_3x3_dw_1x1_stride1_residual<CFG_BLOCK_10_PIC, CFG_BLOCK_10_PC, CFG_BLOCK_10_POC, CFG_BLOCK_10_IC, CFG_BLOCK_10_C, CFG_BLOCK_10_H, CFG_BLOCK_10_W, CFG_BLOCK_10_PW0, CFG_BLOCK_10_PW1, CFG_BLOCK_10_PW2, CFG_BLOCK_10_SW0, CFG_BLOCK_10_SW1, CFG_BLOCK_10_SW2, CFG_BLOCK_10_BW0, CFG_BLOCK_10_BW1, CFG_BLOCK_10_BW2, CFG_BLOCK_10_IW, CFG_AW, CFG_WW, CFG_EXP>(a_block_9, a_block_10, t_block_9, t_block_10, w_block_10_0, w_block_10_1, w_block_10_2, s_block_10_0, s_block_10_1, s_block_10_2, i_block_10);
    conv_1x1_3x3_dw_1x1_stride1<CFG_BLOCK_11_PIC, CFG_BLOCK_11_PC, CFG_BLOCK_11_POC, CFG_BLOCK_11_IC, CFG_BLOCK_11_C, CFG_BLOCK_11_OC, CFG_BLOCK_11_H, CFG_BLOCK_11_W, CFG_BLOCK_11_PW0, CFG_BLOCK_11_PW1, CFG_BLOCK_11_PW2, CFG_BLOCK_11_SW0, CFG_BLOCK_11_SW1, CFG_BLOCK_11_SW2, CFG_BLOCK_11_BW0, CFG_BLOCK_11_BW1, CFG_BLOCK_11_BW2, CFG_AW, CFG_WW, CFG_EXP>(a_block_10, a_block_11, t_block_10, t_block_11, w_block_11_0, w_block_11_1, w_block_11_2, s_block_11_0, s_block_11_1, s_block_11_2);
    conv_1x1_3x3_dw_1x1_stride1<CFG_BLOCK_12_PIC, CFG_BLOCK_12_PC, CFG_BLOCK_12_POC, CFG_BLOCK_12_IC, CFG_BLOCK_12_C, CFG_BLOCK_12_OC, CFG_BLOCK_12_H, CFG_BLOCK_12_W, CFG_BLOCK_12_PW0, CFG_BLOCK_12_PW1, CFG_BLOCK_12_PW2, CFG_BLOCK_12_SW0, CFG_BLOCK_12_SW1, CFG_BLOCK_12_SW2, CFG_BLOCK_12_BW0, CFG_BLOCK_12_BW1, CFG_BLOCK_12_BW2, CFG_AW, CFG_WW, CFG_EXP>(a_block_11, a_block_12, t_block_11, t_block_12, w_block_12_0, w_block_12_1, w_block_12_2, s_block_12_0, s_block_12_1, s_block_12_2);
    conv_1x1_3x3_dw_1x1_stride1_residual<CFG_BLOCK_13_PIC, CFG_BLOCK_13_PC, CFG_BLOCK_13_POC, CFG_BLOCK_13_IC, CFG_BLOCK_13_C, CFG_BLOCK_13_H, CFG_BLOCK_13_W, CFG_BLOCK_13_PW0, CFG_BLOCK_13_PW1, CFG_BLOCK_13_PW2, CFG_BLOCK_13_SW0, CFG_BLOCK_13_SW1, CFG_BLOCK_13_SW2, CFG_BLOCK_13_BW0, CFG_BLOCK_13_BW1, CFG_BLOCK_13_BW2, CFG_BLOCK_13_IW, CFG_AW, CFG_WW, CFG_EXP>(a_block_12, a_block_13, t_block_12, t_block_13, w_block_13_0, w_block_13_1, w_block_13_2, s_block_13_0, s_block_13_1, s_block_13_2, i_block_13);
    conv8<CFG_CONV8_PIC, CFG_CONV8_POC, CFG_CONV8_IC, CFG_CONV8_OC, CFG_CONV8_H, CFG_CONV8_W, CFG_CONV8_PW, CFG_CONV8_SW, CFG_CONV8_BW, CFG_AW, CFG_WW, CFG_EXP>(a_block_13, a_conv8, t_block_13, t_conv8, w_conv8, s_conv8);
    global_avgpool<CFG_FC_PIC, CFG_FC_POC, CFG_FC_IC, CFG_FC_H, CFG_FC_W, CFG_AW>(a_conv8, t_conv8, act_out);

    /*gen_code-store*/
}

void top(ap_int<CFG_AW * CFG_TOP_PIC> *act_in, ap_int<32> *act_out,
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
