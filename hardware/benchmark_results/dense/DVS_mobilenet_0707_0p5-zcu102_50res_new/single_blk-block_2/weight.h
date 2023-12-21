#ifndef __WEIGHT_H__
#define __WEIGHT_H__

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

#endif
