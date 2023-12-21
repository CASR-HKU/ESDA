#ifndef __WEIGHT_H__
#define __WEIGHT_H__

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

#endif
