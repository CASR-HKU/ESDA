#ifndef __WEIGHT_H__
#define __WEIGHT_H__

// BLOCK_4
const ap_int<CFG_BLOCK_4_PIC*CFG_WW> w_block_4_0[CFG_BLOCK_4_C][CFG_BLOCK_4_IC/CFG_BLOCK_4_PIC] = {
    #include "data/block_4_0_w.txt"
};
const ap_int<CFG_BLOCK_4_PC*CFG_WW> w_block_4_1[9][CFG_BLOCK_4_C/CFG_BLOCK_4_PC] = {
    #include "data/block_4_1_w.txt"
};
const ap_int<CFG_BLOCK_4_PC*CFG_WW> w_block_4_2[CFG_BLOCK_4_OC][CFG_BLOCK_4_C/CFG_BLOCK_4_PC] = {
    #include "data/block_4_2_w.txt"
};
const ap_int<CFG_BLOCK_4_SW0+CFG_BLOCK_4_BW0> s_block_4_0[CFG_BLOCK_4_C] = {
    #include "data/block_4_0_s.txt"
};
const ap_int<CFG_BLOCK_4_SW1+CFG_BLOCK_4_BW1> s_block_4_1[CFG_BLOCK_4_C] = {
    #include "data/block_4_1_s.txt"
};
const ap_int<CFG_BLOCK_4_SW2+CFG_BLOCK_4_BW2> s_block_4_2[CFG_BLOCK_4_OC] = {
    #include "data/block_4_2_s.txt"
};
const ap_int<CFG_BLOCK_4_IW> i_block_4 = {
    #include "data/block_4_i.txt"
};

#endif
