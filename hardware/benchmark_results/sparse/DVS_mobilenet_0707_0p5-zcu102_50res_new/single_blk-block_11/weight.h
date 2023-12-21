#ifndef __WEIGHT_H__
#define __WEIGHT_H__

// BLOCK_11
const ap_int<CFG_BLOCK_11_PIC*CFG_WW> w_block_11_0[CFG_BLOCK_11_C][CFG_BLOCK_11_IC/CFG_BLOCK_11_PIC] = {
    #include "data/block_11_0_w.txt"
};
const ap_int<CFG_BLOCK_11_PC*CFG_WW> w_block_11_1[9][CFG_BLOCK_11_C/CFG_BLOCK_11_PC] = {
    #include "data/block_11_1_w.txt"
};
const ap_int<CFG_BLOCK_11_PC*CFG_WW> w_block_11_2[CFG_BLOCK_11_OC][CFG_BLOCK_11_C/CFG_BLOCK_11_PC] = {
    #include "data/block_11_2_w.txt"
};
const ap_int<CFG_BLOCK_11_SW0+CFG_BLOCK_11_BW0> s_block_11_0[CFG_BLOCK_11_C] = {
    #include "data/block_11_0_s.txt"
};
const ap_int<CFG_BLOCK_11_SW1+CFG_BLOCK_11_BW1> s_block_11_1[CFG_BLOCK_11_C] = {
    #include "data/block_11_1_s.txt"
};
const ap_int<CFG_BLOCK_11_SW2+CFG_BLOCK_11_BW2> s_block_11_2[CFG_BLOCK_11_OC] = {
    #include "data/block_11_2_s.txt"
};
const ap_int<CFG_BLOCK_11_IW> i_block_11 = {
    #include "data/block_11_i.txt"
};

#endif
