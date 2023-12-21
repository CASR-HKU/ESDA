#ifndef __WEIGHT_H__
#define __WEIGHT_H__

// BLOCK_5
const ap_int<CFG_BLOCK_5_PIC*CFG_WW> w_block_5_0[CFG_BLOCK_5_C][CFG_BLOCK_5_IC/CFG_BLOCK_5_PIC] = {
    #include "data/block_5_0_w.txt"
};
const ap_int<CFG_BLOCK_5_PC*CFG_WW> w_block_5_1[9][CFG_BLOCK_5_C/CFG_BLOCK_5_PC] = {
    #include "data/block_5_1_w.txt"
};
const ap_int<CFG_BLOCK_5_PC*CFG_WW> w_block_5_2[CFG_BLOCK_5_OC][CFG_BLOCK_5_C/CFG_BLOCK_5_PC] = {
    #include "data/block_5_2_w.txt"
};
const ap_int<CFG_BLOCK_5_SW0+CFG_BLOCK_5_BW0> s_block_5_0[CFG_BLOCK_5_C] = {
    #include "data/block_5_0_s.txt"
};
const ap_int<CFG_BLOCK_5_SW1+CFG_BLOCK_5_BW1> s_block_5_1[CFG_BLOCK_5_C] = {
    #include "data/block_5_1_s.txt"
};
const ap_int<CFG_BLOCK_5_SW2+CFG_BLOCK_5_BW2> s_block_5_2[CFG_BLOCK_5_OC] = {
    #include "data/block_5_2_s.txt"
};
const ap_int<CFG_BLOCK_5_IW> i_block_5 = {
    #include "data/block_5_i.txt"
};

#endif
