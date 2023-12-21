#ifndef __WEIGHT_H__
#define __WEIGHT_H__

// BLOCK_13
const ap_int<CFG_BLOCK_13_PIC*CFG_WW> w_block_13_0[CFG_BLOCK_13_C][CFG_BLOCK_13_IC/CFG_BLOCK_13_PIC] = {
    #include "data/block_13_0_w.txt"
};
const ap_int<CFG_BLOCK_13_PC*CFG_WW> w_block_13_1[9][CFG_BLOCK_13_C/CFG_BLOCK_13_PC] = {
    #include "data/block_13_1_w.txt"
};
const ap_int<CFG_BLOCK_13_PC*CFG_WW> w_block_13_2[CFG_BLOCK_13_OC][CFG_BLOCK_13_C/CFG_BLOCK_13_PC] = {
    #include "data/block_13_2_w.txt"
};
const ap_int<CFG_BLOCK_13_SW0+CFG_BLOCK_13_BW0> s_block_13_0[CFG_BLOCK_13_C] = {
    #include "data/block_13_0_s.txt"
};
const ap_int<CFG_BLOCK_13_SW1+CFG_BLOCK_13_BW1> s_block_13_1[CFG_BLOCK_13_C] = {
    #include "data/block_13_1_s.txt"
};
const ap_int<CFG_BLOCK_13_SW2+CFG_BLOCK_13_BW2> s_block_13_2[CFG_BLOCK_13_OC] = {
    #include "data/block_13_2_s.txt"
};

#endif
