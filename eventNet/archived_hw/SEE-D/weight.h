#ifndef __WEIGHT_H__
#define __WEIGHT_H__

// CONV1
const ap_int<CFG_CONV1_PIC*CFG_WW> w_conv1[9][CFG_CONV1_OC] = {
    #include "data/conv1_w.txt"
};
const ap_int<CFG_CONV1_SW+CFG_CONV1_BW> s_conv1[CFG_CONV1_OC] = {
    #include "data/conv1_s.txt"
};

// BLOCK_0
const ap_int<CFG_BLOCK_0_PIC*CFG_WW> w_block_0_0[CFG_BLOCK_0_C][CFG_BLOCK_0_IC/CFG_BLOCK_0_PIC] = {
    #include "data/block_0_0_w.txt"
};
const ap_int<CFG_BLOCK_0_PC*CFG_WW> w_block_0_1[9][CFG_BLOCK_0_C/CFG_BLOCK_0_PC] = {
    #include "data/block_0_1_w.txt"
};
const ap_int<CFG_BLOCK_0_PC*CFG_WW> w_block_0_2[CFG_BLOCK_0_OC][CFG_BLOCK_0_C/CFG_BLOCK_0_PC] = {
    #include "data/block_0_2_w.txt"
};
const ap_int<CFG_BLOCK_0_SW0+CFG_BLOCK_0_BW0> s_block_0_0[CFG_BLOCK_0_C] = {
    #include "data/block_0_0_s.txt"
};
const ap_int<CFG_BLOCK_0_SW1+CFG_BLOCK_0_BW1> s_block_0_1[CFG_BLOCK_0_C] = {
    #include "data/block_0_1_s.txt"
};
const ap_int<CFG_BLOCK_0_SW2+CFG_BLOCK_0_BW2> s_block_0_2[CFG_BLOCK_0_OC] = {
    #include "data/block_0_2_s.txt"
};

// BLOCK_1
const ap_int<CFG_BLOCK_1_PIC*CFG_WW> w_block_1_0[CFG_BLOCK_1_C][CFG_BLOCK_1_IC/CFG_BLOCK_1_PIC] = {
    #include "data/block_1_0_w.txt"
};
const ap_int<CFG_BLOCK_1_PC*CFG_WW> w_block_1_1[9][CFG_BLOCK_1_C/CFG_BLOCK_1_PC] = {
    #include "data/block_1_1_w.txt"
};
const ap_int<CFG_BLOCK_1_PC*CFG_WW> w_block_1_2[CFG_BLOCK_1_OC][CFG_BLOCK_1_C/CFG_BLOCK_1_PC] = {
    #include "data/block_1_2_w.txt"
};
const ap_int<CFG_BLOCK_1_SW0+CFG_BLOCK_1_BW0> s_block_1_0[CFG_BLOCK_1_C] = {
    #include "data/block_1_0_s.txt"
};
const ap_int<CFG_BLOCK_1_SW1+CFG_BLOCK_1_BW1> s_block_1_1[CFG_BLOCK_1_C] = {
    #include "data/block_1_1_s.txt"
};
const ap_int<CFG_BLOCK_1_SW2+CFG_BLOCK_1_BW2> s_block_1_2[CFG_BLOCK_1_OC] = {
    #include "data/block_1_2_s.txt"
};

// BLOCK_2
const ap_int<CFG_BLOCK_2_PIC*CFG_WW> w_block_2_0[CFG_BLOCK_2_C][CFG_BLOCK_2_IC/CFG_BLOCK_2_PIC] = {
    #include "data/block_2_0_w.txt"
};
const ap_int<CFG_BLOCK_2_PC*CFG_WW> w_block_2_1[9][CFG_BLOCK_2_C/CFG_BLOCK_2_PC] = {
    #include "data/block_2_1_w.txt"
};
const ap_int<CFG_BLOCK_2_PC*CFG_WW> w_block_2_2[CFG_BLOCK_2_OC][CFG_BLOCK_2_C/CFG_BLOCK_2_PC] = {
    #include "data/block_2_2_w.txt"
};
const ap_int<CFG_BLOCK_2_SW0+CFG_BLOCK_2_BW0> s_block_2_0[CFG_BLOCK_2_C] = {
    #include "data/block_2_0_s.txt"
};
const ap_int<CFG_BLOCK_2_SW1+CFG_BLOCK_2_BW1> s_block_2_1[CFG_BLOCK_2_C] = {
    #include "data/block_2_1_s.txt"
};
const ap_int<CFG_BLOCK_2_SW2+CFG_BLOCK_2_BW2> s_block_2_2[CFG_BLOCK_2_OC] = {
    #include "data/block_2_2_s.txt"
};
const ap_int<CFG_BLOCK_2_IW> i_block_2 = {
    #include "data/block_2_i.txt"
};

// BLOCK_3
const ap_int<CFG_BLOCK_3_PIC*CFG_WW> w_block_3_0[CFG_BLOCK_3_C][CFG_BLOCK_3_IC/CFG_BLOCK_3_PIC] = {
    #include "data/block_3_0_w.txt"
};
const ap_int<CFG_BLOCK_3_PC*CFG_WW> w_block_3_1[9][CFG_BLOCK_3_C/CFG_BLOCK_3_PC] = {
    #include "data/block_3_1_w.txt"
};
const ap_int<CFG_BLOCK_3_PC*CFG_WW> w_block_3_2[CFG_BLOCK_3_OC][CFG_BLOCK_3_C/CFG_BLOCK_3_PC] = {
    #include "data/block_3_2_w.txt"
};
const ap_int<CFG_BLOCK_3_SW0+CFG_BLOCK_3_BW0> s_block_3_0[CFG_BLOCK_3_C] = {
    #include "data/block_3_0_s.txt"
};
const ap_int<CFG_BLOCK_3_SW1+CFG_BLOCK_3_BW1> s_block_3_1[CFG_BLOCK_3_C] = {
    #include "data/block_3_1_s.txt"
};
const ap_int<CFG_BLOCK_3_SW2+CFG_BLOCK_3_BW2> s_block_3_2[CFG_BLOCK_3_OC] = {
    #include "data/block_3_2_s.txt"
};

// BLOCK_4
const ap_int<CFG_BLOCK_4_PIC*CFG_WW> w_block_4_0[CFG_BLOCK_4_C][CFG_BLOCK_4_IC/CFG_BLOCK_4_PIC] = {
    #include "data/block_4_0_w.txt"
};
const ap_int<CFG_BLOCK_4_PC*CFG_WW> w_block_4_1[9][CFG_BLOCK_4_C/CFG_BLOCK_4_PC] = {
    #include "data/block_4_1_w.txt"
};
const ap_int<CFG_BLOCK_4_PC*CFG_WW> w_block_4_2[CFG_BLOCK_4_OC][CFG_BLOCK_4_C/CFG_BLOCK_4_PC] = {
    #include "data/block_4_2_w.txt"
};
const ap_int<CFG_BLOCK_4_SW0+CFG_BLOCK_4_BW0> s_block_4_0[CFG_BLOCK_4_C] = {
    #include "data/block_4_0_s.txt"
};
const ap_int<CFG_BLOCK_4_SW1+CFG_BLOCK_4_BW1> s_block_4_1[CFG_BLOCK_4_C] = {
    #include "data/block_4_1_s.txt"
};
const ap_int<CFG_BLOCK_4_SW2+CFG_BLOCK_4_BW2> s_block_4_2[CFG_BLOCK_4_OC] = {
    #include "data/block_4_2_s.txt"
};
const ap_int<CFG_BLOCK_4_IW> i_block_4 = {
    #include "data/block_4_i.txt"
};

// BLOCK_5
const ap_int<CFG_BLOCK_5_PIC*CFG_WW> w_block_5_0[CFG_BLOCK_5_C][CFG_BLOCK_5_IC/CFG_BLOCK_5_PIC] = {
    #include "data/block_5_0_w.txt"
};
const ap_int<CFG_BLOCK_5_PC*CFG_WW> w_block_5_1[9][CFG_BLOCK_5_C/CFG_BLOCK_5_PC] = {
    #include "data/block_5_1_w.txt"
};
const ap_int<CFG_BLOCK_5_PC*CFG_WW> w_block_5_2[CFG_BLOCK_5_OC][CFG_BLOCK_5_C/CFG_BLOCK_5_PC] = {
    #include "data/block_5_2_w.txt"
};
const ap_int<CFG_BLOCK_5_SW0+CFG_BLOCK_5_BW0> s_block_5_0[CFG_BLOCK_5_C] = {
    #include "data/block_5_0_s.txt"
};
const ap_int<CFG_BLOCK_5_SW1+CFG_BLOCK_5_BW1> s_block_5_1[CFG_BLOCK_5_C] = {
    #include "data/block_5_1_s.txt"
};
const ap_int<CFG_BLOCK_5_SW2+CFG_BLOCK_5_BW2> s_block_5_2[CFG_BLOCK_5_OC] = {
    #include "data/block_5_2_s.txt"
};

// BLOCK_6
const ap_int<CFG_BLOCK_6_PIC*CFG_WW> w_block_6_0[CFG_BLOCK_6_C][CFG_BLOCK_6_IC/CFG_BLOCK_6_PIC] = {
    #include "data/block_6_0_w.txt"
};
const ap_int<CFG_BLOCK_6_PC*CFG_WW> w_block_6_1[9][CFG_BLOCK_6_C/CFG_BLOCK_6_PC] = {
    #include "data/block_6_1_w.txt"
};
const ap_int<CFG_BLOCK_6_PC*CFG_WW> w_block_6_2[CFG_BLOCK_6_OC][CFG_BLOCK_6_C/CFG_BLOCK_6_PC] = {
    #include "data/block_6_2_w.txt"
};
const ap_int<CFG_BLOCK_6_SW0+CFG_BLOCK_6_BW0> s_block_6_0[CFG_BLOCK_6_C] = {
    #include "data/block_6_0_s.txt"
};
const ap_int<CFG_BLOCK_6_SW1+CFG_BLOCK_6_BW1> s_block_6_1[CFG_BLOCK_6_C] = {
    #include "data/block_6_1_s.txt"
};
const ap_int<CFG_BLOCK_6_SW2+CFG_BLOCK_6_BW2> s_block_6_2[CFG_BLOCK_6_OC] = {
    #include "data/block_6_2_s.txt"
};
const ap_int<CFG_BLOCK_6_IW> i_block_6 = {
    #include "data/block_6_i.txt"
};

// BLOCK_7
const ap_int<CFG_BLOCK_7_PIC*CFG_WW> w_block_7_0[CFG_BLOCK_7_C][CFG_BLOCK_7_IC/CFG_BLOCK_7_PIC] = {
    #include "data/block_7_0_w.txt"
};
const ap_int<CFG_BLOCK_7_PC*CFG_WW> w_block_7_1[9][CFG_BLOCK_7_C/CFG_BLOCK_7_PC] = {
    #include "data/block_7_1_w.txt"
};
const ap_int<CFG_BLOCK_7_PC*CFG_WW> w_block_7_2[CFG_BLOCK_7_OC][CFG_BLOCK_7_C/CFG_BLOCK_7_PC] = {
    #include "data/block_7_2_w.txt"
};
const ap_int<CFG_BLOCK_7_SW0+CFG_BLOCK_7_BW0> s_block_7_0[CFG_BLOCK_7_C] = {
    #include "data/block_7_0_s.txt"
};
const ap_int<CFG_BLOCK_7_SW1+CFG_BLOCK_7_BW1> s_block_7_1[CFG_BLOCK_7_C] = {
    #include "data/block_7_1_s.txt"
};
const ap_int<CFG_BLOCK_7_SW2+CFG_BLOCK_7_BW2> s_block_7_2[CFG_BLOCK_7_OC] = {
    #include "data/block_7_2_s.txt"
};
const ap_int<CFG_BLOCK_7_IW> i_block_7 = {
    #include "data/block_7_i.txt"
};

// BLOCK_8
const ap_int<CFG_BLOCK_8_PIC*CFG_WW> w_block_8_0[CFG_BLOCK_8_C][CFG_BLOCK_8_IC/CFG_BLOCK_8_PIC] = {
    #include "data/block_8_0_w.txt"
};
const ap_int<CFG_BLOCK_8_PC*CFG_WW> w_block_8_1[9][CFG_BLOCK_8_C/CFG_BLOCK_8_PC] = {
    #include "data/block_8_1_w.txt"
};
const ap_int<CFG_BLOCK_8_PC*CFG_WW> w_block_8_2[CFG_BLOCK_8_OC][CFG_BLOCK_8_C/CFG_BLOCK_8_PC] = {
    #include "data/block_8_2_w.txt"
};
const ap_int<CFG_BLOCK_8_SW0+CFG_BLOCK_8_BW0> s_block_8_0[CFG_BLOCK_8_C] = {
    #include "data/block_8_0_s.txt"
};
const ap_int<CFG_BLOCK_8_SW1+CFG_BLOCK_8_BW1> s_block_8_1[CFG_BLOCK_8_C] = {
    #include "data/block_8_1_s.txt"
};
const ap_int<CFG_BLOCK_8_SW2+CFG_BLOCK_8_BW2> s_block_8_2[CFG_BLOCK_8_OC] = {
    #include "data/block_8_2_s.txt"
};

// CONV8
const ap_int<CFG_CONV8_PIC*CFG_WW> w_conv8[CFG_CONV8_OC][CFG_CONV8_IC/CFG_CONV8_PIC] = {
    #include "data/conv8_w.txt"
};
const ap_int<CFG_CONV8_SW+CFG_CONV8_BW> s_conv8[CFG_CONV8_OC] = {
    #include "data/conv8_s.txt"
};

// FC

#endif
