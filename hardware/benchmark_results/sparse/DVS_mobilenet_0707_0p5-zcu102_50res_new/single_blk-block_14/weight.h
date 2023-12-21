#ifndef __WEIGHT_H__
#define __WEIGHT_H__

// BLOCK_14
const ap_int<CFG_BLOCK_14_PIC*CFG_WW> w_block_14_0[CFG_BLOCK_14_C][CFG_BLOCK_14_IC/CFG_BLOCK_14_PIC] = {
    #include "data/block_14_0_w.txt"
};
const ap_int<CFG_BLOCK_14_PC*CFG_WW> w_block_14_1[9][CFG_BLOCK_14_C/CFG_BLOCK_14_PC] = {
    #include "data/block_14_1_w.txt"
};
const ap_int<CFG_BLOCK_14_PC*CFG_WW> w_block_14_2[CFG_BLOCK_14_OC][CFG_BLOCK_14_C/CFG_BLOCK_14_PC] = {
    #include "data/block_14_2_w.txt"
};
const ap_int<CFG_BLOCK_14_SW0+CFG_BLOCK_14_BW0> s_block_14_0[CFG_BLOCK_14_C] = {
    #include "data/block_14_0_s.txt"
};
const ap_int<CFG_BLOCK_14_SW1+CFG_BLOCK_14_BW1> s_block_14_1[CFG_BLOCK_14_C] = {
    #include "data/block_14_1_s.txt"
};
const ap_int<CFG_BLOCK_14_SW2+CFG_BLOCK_14_BW2> s_block_14_2[CFG_BLOCK_14_OC] = {
    #include "data/block_14_2_s.txt"
};
const ap_int<CFG_BLOCK_14_IW> i_block_14 = {
    #include "data/block_14_i.txt"
};

#endif
