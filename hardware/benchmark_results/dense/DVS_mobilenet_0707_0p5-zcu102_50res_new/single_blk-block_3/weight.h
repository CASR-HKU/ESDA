#ifndef __WEIGHT_H__
#define __WEIGHT_H__

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

#endif
