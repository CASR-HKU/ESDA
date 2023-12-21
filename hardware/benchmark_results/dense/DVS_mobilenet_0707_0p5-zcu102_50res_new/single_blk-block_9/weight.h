#ifndef __WEIGHT_H__
#define __WEIGHT_H__

// BLOCK_9
const ap_int<CFG_BLOCK_9_PIC*CFG_WW> w_block_9_0[CFG_BLOCK_9_C][CFG_BLOCK_9_IC/CFG_BLOCK_9_PIC] = {
    #include "data/block_9_0_w.txt"
};
const ap_int<CFG_BLOCK_9_PC*CFG_WW> w_block_9_1[9][CFG_BLOCK_9_C/CFG_BLOCK_9_PC] = {
    #include "data/block_9_1_w.txt"
};
const ap_int<CFG_BLOCK_9_PC*CFG_WW> w_block_9_2[CFG_BLOCK_9_OC][CFG_BLOCK_9_C/CFG_BLOCK_9_PC] = {
    #include "data/block_9_2_w.txt"
};
const ap_int<CFG_BLOCK_9_SW0+CFG_BLOCK_9_BW0> s_block_9_0[CFG_BLOCK_9_C] = {
    #include "data/block_9_0_s.txt"
};
const ap_int<CFG_BLOCK_9_SW1+CFG_BLOCK_9_BW1> s_block_9_1[CFG_BLOCK_9_C] = {
    #include "data/block_9_1_s.txt"
};
const ap_int<CFG_BLOCK_9_SW2+CFG_BLOCK_9_BW2> s_block_9_2[CFG_BLOCK_9_OC] = {
    #include "data/block_9_2_s.txt"
};
const ap_int<CFG_BLOCK_9_IW> i_block_9 = {
    #include "data/block_9_i.txt"
};

#endif
