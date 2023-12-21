#ifndef __WEIGHT_H__
#define __WEIGHT_H__

// BLOCK_16
const ap_int<CFG_BLOCK_16_PIC*CFG_WW> w_block_16_0[CFG_BLOCK_16_C][CFG_BLOCK_16_IC/CFG_BLOCK_16_PIC] = {
    #include "data/block_16_0_w.txt"
};
const ap_int<CFG_BLOCK_16_PC*CFG_WW> w_block_16_1[9][CFG_BLOCK_16_C/CFG_BLOCK_16_PC] = {
    #include "data/block_16_1_w.txt"
};
const ap_int<CFG_BLOCK_16_PC*CFG_WW> w_block_16_2[CFG_BLOCK_16_OC][CFG_BLOCK_16_C/CFG_BLOCK_16_PC] = {
    #include "data/block_16_2_w.txt"
};
const ap_int<CFG_BLOCK_16_SW0+CFG_BLOCK_16_BW0> s_block_16_0[CFG_BLOCK_16_C] = {
    #include "data/block_16_0_s.txt"
};
const ap_int<CFG_BLOCK_16_SW1+CFG_BLOCK_16_BW1> s_block_16_1[CFG_BLOCK_16_C] = {
    #include "data/block_16_1_s.txt"
};
const ap_int<CFG_BLOCK_16_SW2+CFG_BLOCK_16_BW2> s_block_16_2[CFG_BLOCK_16_OC] = {
    #include "data/block_16_2_s.txt"
};

#endif
