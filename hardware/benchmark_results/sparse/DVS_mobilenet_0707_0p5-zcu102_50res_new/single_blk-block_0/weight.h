#ifndef __WEIGHT_H__
#define __WEIGHT_H__

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

#endif
