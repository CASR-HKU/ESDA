#ifndef __WEIGHT_H__
#define __WEIGHT_H__

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

#endif
