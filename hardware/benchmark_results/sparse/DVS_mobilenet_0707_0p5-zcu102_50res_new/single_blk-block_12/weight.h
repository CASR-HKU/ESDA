#ifndef __WEIGHT_H__
#define __WEIGHT_H__

// BLOCK_12
const ap_int<CFG_BLOCK_12_PIC*CFG_WW> w_block_12_0[CFG_BLOCK_12_C][CFG_BLOCK_12_IC/CFG_BLOCK_12_PIC] = {
    #include "data/block_12_0_w.txt"
};
const ap_int<CFG_BLOCK_12_PC*CFG_WW> w_block_12_1[9][CFG_BLOCK_12_C/CFG_BLOCK_12_PC] = {
    #include "data/block_12_1_w.txt"
};
const ap_int<CFG_BLOCK_12_PC*CFG_WW> w_block_12_2[CFG_BLOCK_12_OC][CFG_BLOCK_12_C/CFG_BLOCK_12_PC] = {
    #include "data/block_12_2_w.txt"
};
const ap_int<CFG_BLOCK_12_SW0+CFG_BLOCK_12_BW0> s_block_12_0[CFG_BLOCK_12_C] = {
    #include "data/block_12_0_s.txt"
};
const ap_int<CFG_BLOCK_12_SW1+CFG_BLOCK_12_BW1> s_block_12_1[CFG_BLOCK_12_C] = {
    #include "data/block_12_1_s.txt"
};
const ap_int<CFG_BLOCK_12_SW2+CFG_BLOCK_12_BW2> s_block_12_2[CFG_BLOCK_12_OC] = {
    #include "data/block_12_2_s.txt"
};
const ap_int<CFG_BLOCK_12_IW> i_block_12 = {
    #include "data/block_12_i.txt"
};

#endif
