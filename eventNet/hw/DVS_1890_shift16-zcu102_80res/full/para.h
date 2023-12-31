#ifndef __PARA_H__
#define __PARA_H__

// COMMON
#define CFG_AW 8   // AW
#define CFG_WW 8   // WW
#define CFG_PW 32  // PSUMW
#define CFG_SW 16  // SCALEW
#define CFG_BW 16  // BIASW
#define CFG_TW 8   // TOKENW
#define CFG_MW 128  // to be very carefull
#define CFG_EXP 16 // EXP

// CONV1
#define CFG_CONV1_PIC 2
#define CFG_CONV1_POC 8
#define CFG_CONV1_IC 2
#define CFG_CONV1_OC 32
#define CFG_CONV1_H 128
#define CFG_CONV1_W 128
#define CFG_CONV1_PW (CFG_AW + CFG_WW + 1)
#define CFG_CONV1_SW CFG_SW
#define CFG_CONV1_BW CFG_BW

// BLOCK_0
#define CFG_BLOCK_0_PIC 8
#define CFG_BLOCK_0_PC 8
#define CFG_BLOCK_0_POC 8
#define CFG_BLOCK_0_IC 32
#define CFG_BLOCK_0_C 32
#define CFG_BLOCK_0_OC 16
#define CFG_BLOCK_0_H 64
#define CFG_BLOCK_0_W 64
#define CFG_BLOCK_0_PW0 (CFG_AW + CFG_WW + 5)
#define CFG_BLOCK_0_PW1 (CFG_AW + CFG_WW + 4)
#define CFG_BLOCK_0_PW2 (CFG_AW + CFG_WW + 4)
#define CFG_BLOCK_0_SW0 CFG_SW
#define CFG_BLOCK_0_SW1 CFG_SW
#define CFG_BLOCK_0_SW2 CFG_SW
#define CFG_BLOCK_0_BW0 CFG_BW
#define CFG_BLOCK_0_BW1 CFG_BW
#define CFG_BLOCK_0_BW2 CFG_BW

// BLOCK_1
#define CFG_BLOCK_1_PIC 8
#define CFG_BLOCK_1_PC 8
#define CFG_BLOCK_1_POC 4
#define CFG_BLOCK_1_IC 16
#define CFG_BLOCK_1_C 16
#define CFG_BLOCK_1_OC 16
#define CFG_BLOCK_1_H 32
#define CFG_BLOCK_1_W 32
#define CFG_BLOCK_1_PW0 (CFG_AW + CFG_WW + 4)
#define CFG_BLOCK_1_PW1 (CFG_AW + CFG_WW + 4)
#define CFG_BLOCK_1_PW2 (CFG_AW + CFG_WW + 4)
#define CFG_BLOCK_1_SW0 CFG_SW
#define CFG_BLOCK_1_SW1 CFG_SW
#define CFG_BLOCK_1_SW2 CFG_SW
#define CFG_BLOCK_1_BW0 CFG_BW
#define CFG_BLOCK_1_BW1 CFG_BW
#define CFG_BLOCK_1_BW2 CFG_BW
#define CFG_BLOCK_1_IW (CFG_SW + 1)

// BLOCK_2
#define CFG_BLOCK_2_PIC 4
#define CFG_BLOCK_2_PC 8
#define CFG_BLOCK_2_POC 4
#define CFG_BLOCK_2_IC 16
#define CFG_BLOCK_2_C 16
#define CFG_BLOCK_2_OC 16
#define CFG_BLOCK_2_H 32
#define CFG_BLOCK_2_W 32
#define CFG_BLOCK_2_PW0 (CFG_AW + CFG_WW + 4)
#define CFG_BLOCK_2_PW1 (CFG_AW + CFG_WW + 4)
#define CFG_BLOCK_2_PW2 (CFG_AW + CFG_WW + 4)
#define CFG_BLOCK_2_SW0 CFG_SW
#define CFG_BLOCK_2_SW1 CFG_SW
#define CFG_BLOCK_2_SW2 CFG_SW
#define CFG_BLOCK_2_BW0 CFG_BW
#define CFG_BLOCK_2_BW1 CFG_BW
#define CFG_BLOCK_2_BW2 CFG_BW
#define CFG_BLOCK_2_IW (CFG_SW + 1)

// BLOCK_3
#define CFG_BLOCK_3_PIC 4
#define CFG_BLOCK_3_PC 16
#define CFG_BLOCK_3_POC 8
#define CFG_BLOCK_3_IC 16
#define CFG_BLOCK_3_C 64
#define CFG_BLOCK_3_OC 32
#define CFG_BLOCK_3_H 32
#define CFG_BLOCK_3_W 32
#define CFG_BLOCK_3_PW0 (CFG_AW + CFG_WW + 4)
#define CFG_BLOCK_3_PW1 (CFG_AW + CFG_WW + 4)
#define CFG_BLOCK_3_PW2 (CFG_AW + CFG_WW + 5)
#define CFG_BLOCK_3_SW0 CFG_SW
#define CFG_BLOCK_3_SW1 CFG_SW
#define CFG_BLOCK_3_SW2 CFG_SW
#define CFG_BLOCK_3_BW0 CFG_BW
#define CFG_BLOCK_3_BW1 CFG_BW
#define CFG_BLOCK_3_BW2 CFG_BW

// BLOCK_4
#define CFG_BLOCK_4_PIC 8
#define CFG_BLOCK_4_PC 8
#define CFG_BLOCK_4_POC 8
#define CFG_BLOCK_4_IC 32
#define CFG_BLOCK_4_C 64
#define CFG_BLOCK_4_OC 48
#define CFG_BLOCK_4_H 16
#define CFG_BLOCK_4_W 16
#define CFG_BLOCK_4_PW0 (CFG_AW + CFG_WW + 5)
#define CFG_BLOCK_4_PW1 (CFG_AW + CFG_WW + 4)
#define CFG_BLOCK_4_PW2 (CFG_AW + CFG_WW + 6)
#define CFG_BLOCK_4_SW0 CFG_SW
#define CFG_BLOCK_4_SW1 CFG_SW
#define CFG_BLOCK_4_SW2 CFG_SW
#define CFG_BLOCK_4_BW0 CFG_BW
#define CFG_BLOCK_4_BW1 CFG_BW
#define CFG_BLOCK_4_BW2 CFG_BW

// BLOCK_5
#define CFG_BLOCK_5_PIC 8
#define CFG_BLOCK_5_PC 16
#define CFG_BLOCK_5_POC 6
#define CFG_BLOCK_5_IC 48
#define CFG_BLOCK_5_C 96
#define CFG_BLOCK_5_OC 48
#define CFG_BLOCK_5_H 8
#define CFG_BLOCK_5_W 8
#define CFG_BLOCK_5_PW0 (CFG_AW + CFG_WW + 6)
#define CFG_BLOCK_5_PW1 (CFG_AW + CFG_WW + 4)
#define CFG_BLOCK_5_PW2 (CFG_AW + CFG_WW + 6)
#define CFG_BLOCK_5_SW0 CFG_SW
#define CFG_BLOCK_5_SW1 CFG_SW
#define CFG_BLOCK_5_SW2 CFG_SW
#define CFG_BLOCK_5_BW0 CFG_BW
#define CFG_BLOCK_5_BW1 CFG_BW
#define CFG_BLOCK_5_BW2 CFG_BW
#define CFG_BLOCK_5_IW (CFG_SW + 1)

// BLOCK_6
#define CFG_BLOCK_6_PIC 6
#define CFG_BLOCK_6_PC 4
#define CFG_BLOCK_6_POC 8
#define CFG_BLOCK_6_IC 48
#define CFG_BLOCK_6_C 48
#define CFG_BLOCK_6_OC 72
#define CFG_BLOCK_6_H 8
#define CFG_BLOCK_6_W 8
#define CFG_BLOCK_6_PW0 (CFG_AW + CFG_WW + 6)
#define CFG_BLOCK_6_PW1 (CFG_AW + CFG_WW + 4)
#define CFG_BLOCK_6_PW2 (CFG_AW + CFG_WW + 7)
#define CFG_BLOCK_6_SW0 CFG_SW
#define CFG_BLOCK_6_SW1 CFG_SW
#define CFG_BLOCK_6_SW2 CFG_SW
#define CFG_BLOCK_6_BW0 CFG_BW
#define CFG_BLOCK_6_BW1 CFG_BW
#define CFG_BLOCK_6_BW2 CFG_BW

// BLOCK_7
#define CFG_BLOCK_7_PIC 8
#define CFG_BLOCK_7_PC 4
#define CFG_BLOCK_7_POC 8
#define CFG_BLOCK_7_IC 72
#define CFG_BLOCK_7_C 72
#define CFG_BLOCK_7_OC 72
#define CFG_BLOCK_7_H 4
#define CFG_BLOCK_7_W 4
#define CFG_BLOCK_7_PW0 (CFG_AW + CFG_WW + 7)
#define CFG_BLOCK_7_PW1 (CFG_AW + CFG_WW + 4)
#define CFG_BLOCK_7_PW2 (CFG_AW + CFG_WW + 7)
#define CFG_BLOCK_7_SW0 CFG_SW
#define CFG_BLOCK_7_SW1 CFG_SW
#define CFG_BLOCK_7_SW2 CFG_SW
#define CFG_BLOCK_7_BW0 CFG_BW
#define CFG_BLOCK_7_BW1 CFG_BW
#define CFG_BLOCK_7_BW2 CFG_BW
#define CFG_BLOCK_7_IW (CFG_SW + 1)

// BLOCK_8
#define CFG_BLOCK_8_PIC 8
#define CFG_BLOCK_8_PC 4
#define CFG_BLOCK_8_POC 8
#define CFG_BLOCK_8_IC 72
#define CFG_BLOCK_8_C 72
#define CFG_BLOCK_8_OC 72
#define CFG_BLOCK_8_H 4
#define CFG_BLOCK_8_W 4
#define CFG_BLOCK_8_PW0 (CFG_AW + CFG_WW + 7)
#define CFG_BLOCK_8_PW1 (CFG_AW + CFG_WW + 4)
#define CFG_BLOCK_8_PW2 (CFG_AW + CFG_WW + 7)
#define CFG_BLOCK_8_SW0 CFG_SW
#define CFG_BLOCK_8_SW1 CFG_SW
#define CFG_BLOCK_8_SW2 CFG_SW
#define CFG_BLOCK_8_BW0 CFG_BW
#define CFG_BLOCK_8_BW1 CFG_BW
#define CFG_BLOCK_8_BW2 CFG_BW
#define CFG_BLOCK_8_IW (CFG_SW + 1)

// BLOCK_9
#define CFG_BLOCK_9_PIC 8
#define CFG_BLOCK_9_PC 8
#define CFG_BLOCK_9_POC 12
#define CFG_BLOCK_9_IC 72
#define CFG_BLOCK_9_C 144
#define CFG_BLOCK_9_OC 96
#define CFG_BLOCK_9_H 4
#define CFG_BLOCK_9_W 4
#define CFG_BLOCK_9_PW0 (CFG_AW + CFG_WW + 7)
#define CFG_BLOCK_9_PW1 (CFG_AW + CFG_WW + 4)
#define CFG_BLOCK_9_PW2 (CFG_AW + CFG_WW + 7)
#define CFG_BLOCK_9_SW0 CFG_SW
#define CFG_BLOCK_9_SW1 CFG_SW
#define CFG_BLOCK_9_SW2 CFG_SW
#define CFG_BLOCK_9_BW0 CFG_BW
#define CFG_BLOCK_9_BW1 CFG_BW
#define CFG_BLOCK_9_BW2 CFG_BW

// BLOCK_10
#define CFG_BLOCK_10_PIC 12
#define CFG_BLOCK_10_PC 12
#define CFG_BLOCK_10_POC 24
#define CFG_BLOCK_10_IC 96
#define CFG_BLOCK_10_C 192
#define CFG_BLOCK_10_OC 96
#define CFG_BLOCK_10_H 4
#define CFG_BLOCK_10_W 4
#define CFG_BLOCK_10_PW0 (CFG_AW + CFG_WW + 7)
#define CFG_BLOCK_10_PW1 (CFG_AW + CFG_WW + 4)
#define CFG_BLOCK_10_PW2 (CFG_AW + CFG_WW + 7)
#define CFG_BLOCK_10_SW0 CFG_SW
#define CFG_BLOCK_10_SW1 CFG_SW
#define CFG_BLOCK_10_SW2 CFG_SW
#define CFG_BLOCK_10_BW0 CFG_BW
#define CFG_BLOCK_10_BW1 CFG_BW
#define CFG_BLOCK_10_BW2 CFG_BW
#define CFG_BLOCK_10_IW (CFG_SW + 1)

// CONV8
#define CFG_CONV8_PIC 24
#define CFG_CONV8_POC 32
#define CFG_CONV8_IC 96
#define CFG_CONV8_OC 1280
#define CFG_CONV8_H 4
#define CFG_CONV8_W 4
#define CFG_CONV8_PW (CFG_AW + CFG_WW + 7)
#define CFG_CONV8_SW CFG_SW
#define CFG_CONV8_BW CFG_BW

// FC
#define CFG_FC_PIC 32
#define CFG_FC_POC 2
#define CFG_FC_IC 1280
#define CFG_FC_OC 10
#define CFG_FC_H 4
#define CFG_FC_W 4

// TOP
#define CFG_TOP_IC CFG_CONV1_IC
#define CFG_TOP_PIC CFG_CONV1_PIC
#define CFG_TOP_IH CFG_CONV1_H
#define CFG_TOP_IW CFG_CONV1_W
#define CFG_TOP_OC CFG_FC_OC
#define CFG_TOP_POC CFG_FC_POC
#define CFG_TOP_OH (CFG_FC_H/1)
#define CFG_TOP_OW (CFG_FC_W/1)

#endif
