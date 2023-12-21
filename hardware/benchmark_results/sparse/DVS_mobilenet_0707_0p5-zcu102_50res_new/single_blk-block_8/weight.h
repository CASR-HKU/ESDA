#ifndef __WEIGHT_H__
#define __WEIGHT_H__

// BLOCK_8
const ap_int<CFG_BLOCK_8_PIC*CFG_WW> w_block_8_0[CFG_BLOCK_8_C][CFG_BLOCK_8_IC/CFG_BLOCK_8_PIC] = {
    #include "data/block_8_0_w.txt"
};
const ap_int<CFG_BLOCK_8_PC*CFG_WW> w_block_8_1[9][CFG_BLOCK_8_C/CFG_BLOCK_8_PC] = {
    #include "data/block_8_1_w.txt"
};
const ap_int<CFG_BLOCK_8_PC*CFG_WW> w_block_8_2[CFG_BLOCK_8_OC][CFG_BLOCK_8_C/CFG_BLOCK_8_PC] = {
    #include "data/block_8_2_w.txt"
};
const ap_int<CFG_BLOCK_8_SW0+CFG_BLOCK_8_BW0> s_block_8_0[CFG_BLOCK_8_C] = {
    #include "data/block_8_0_s.txt"
};
const ap_int<CFG_BLOCK_8_SW1+CFG_BLOCK_8_BW1> s_block_8_1[CFG_BLOCK_8_C] = {
    #include "data/block_8_1_s.txt"
};
const ap_int<CFG_BLOCK_8_SW2+CFG_BLOCK_8_BW2> s_block_8_2[CFG_BLOCK_8_OC] = {
    #include "data/block_8_2_s.txt"
};
const ap_int<CFG_BLOCK_8_IW> i_block_8 = {
    #include "data/block_8_i.txt"
};

#endif
