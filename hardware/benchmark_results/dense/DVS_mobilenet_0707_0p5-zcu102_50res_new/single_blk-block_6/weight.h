#ifndef __WEIGHT_H__
#define __WEIGHT_H__

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

#endif
