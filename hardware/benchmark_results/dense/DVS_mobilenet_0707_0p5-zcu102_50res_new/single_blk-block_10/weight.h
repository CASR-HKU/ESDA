#ifndef __WEIGHT_H__
#define __WEIGHT_H__

// BLOCK_10
const ap_int<CFG_BLOCK_10_PIC*CFG_WW> w_block_10_0[CFG_BLOCK_10_C][CFG_BLOCK_10_IC/CFG_BLOCK_10_PIC] = {
    #include "data/block_10_0_w.txt"
};
const ap_int<CFG_BLOCK_10_PC*CFG_WW> w_block_10_1[9][CFG_BLOCK_10_C/CFG_BLOCK_10_PC] = {
    #include "data/block_10_1_w.txt"
};
const ap_int<CFG_BLOCK_10_PC*CFG_WW> w_block_10_2[CFG_BLOCK_10_OC][CFG_BLOCK_10_C/CFG_BLOCK_10_PC] = {
    #include "data/block_10_2_w.txt"
};
const ap_int<CFG_BLOCK_10_SW0+CFG_BLOCK_10_BW0> s_block_10_0[CFG_BLOCK_10_C] = {
    #include "data/block_10_0_s.txt"
};
const ap_int<CFG_BLOCK_10_SW1+CFG_BLOCK_10_BW1> s_block_10_1[CFG_BLOCK_10_C] = {
    #include "data/block_10_1_s.txt"
};
const ap_int<CFG_BLOCK_10_SW2+CFG_BLOCK_10_BW2> s_block_10_2[CFG_BLOCK_10_OC] = {
    #include "data/block_10_2_s.txt"
};

#endif
