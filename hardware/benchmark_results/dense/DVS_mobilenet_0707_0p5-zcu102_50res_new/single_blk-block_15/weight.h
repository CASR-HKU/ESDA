#ifndef __WEIGHT_H__
#define __WEIGHT_H__

// BLOCK_15
const ap_int<CFG_BLOCK_15_PIC*CFG_WW> w_block_15_0[CFG_BLOCK_15_C][CFG_BLOCK_15_IC/CFG_BLOCK_15_PIC] = {
    #include "data/block_15_0_w.txt"
};
const ap_int<CFG_BLOCK_15_PC*CFG_WW> w_block_15_1[9][CFG_BLOCK_15_C/CFG_BLOCK_15_PC] = {
    #include "data/block_15_1_w.txt"
};
const ap_int<CFG_BLOCK_15_PC*CFG_WW> w_block_15_2[CFG_BLOCK_15_OC][CFG_BLOCK_15_C/CFG_BLOCK_15_PC] = {
    #include "data/block_15_2_w.txt"
};
const ap_int<CFG_BLOCK_15_SW0+CFG_BLOCK_15_BW0> s_block_15_0[CFG_BLOCK_15_C] = {
    #include "data/block_15_0_s.txt"
};
const ap_int<CFG_BLOCK_15_SW1+CFG_BLOCK_15_BW1> s_block_15_1[CFG_BLOCK_15_C] = {
    #include "data/block_15_1_s.txt"
};
const ap_int<CFG_BLOCK_15_SW2+CFG_BLOCK_15_BW2> s_block_15_2[CFG_BLOCK_15_OC] = {
    #include "data/block_15_2_s.txt"
};
const ap_int<CFG_BLOCK_15_IW> i_block_15 = {
    #include "data/block_15_i.txt"
};

#endif
