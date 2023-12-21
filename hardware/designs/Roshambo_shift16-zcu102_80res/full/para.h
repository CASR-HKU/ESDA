#ifndef __PARA_H__
#define __PARA_H__

// COMMON
#define CFG_AW 8   // AW
#define CFG_WW 8   // WW
#define CFG_PW 32  // PSUMW
#define CFG_SW 16  // SCALEW
#define CFG_BW 16  // BIASW
#define CFG_TW 8   // TOKENW
#define CFG_MW 64  // to be very carefull
#define CFG_EXP 16 // EXP

// CONV1
#define CFG_CONV1_PIC 1
#define CFG_CONV1_POC 6
#define CFG_CONV1_IC 1
#define CFG_CONV1_OC 24
#define CFG_CONV1_H 64
#define CFG_CONV1_W 64
#define CFG_CONV1_PW (CFG_AW + CFG_WW + 4)
#define CFG_CONV1_SW CFG_SW
#define CFG_CONV1_BW CFG_BW

// BLOCK_0
#define CFG_BLOCK_0_PIC 6
#define CFG_BLOCK_0_PC 24
#define CFG_BLOCK_0_POC 4
#define CFG_BLOCK_0_IC 24
#define CFG_BLOCK_0_C 24
#define CFG_BLOCK_0_OC 16
#define CFG_BLOCK_0_H 32
#define CFG_BLOCK_0_W 32
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
#define CFG_BLOCK_1_PIC 4
#define CFG_BLOCK_1_PC 16
#define CFG_BLOCK_1_POC 6
#define CFG_BLOCK_1_IC 16
#define CFG_BLOCK_1_C 64
#define CFG_BLOCK_1_OC 24
#define CFG_BLOCK_1_H 32
#define CFG_BLOCK_1_W 32
#define CFG_BLOCK_1_PW0 (CFG_AW + CFG_WW + 4)
#define CFG_BLOCK_1_PW1 (CFG_AW + CFG_WW + 4)
#define CFG_BLOCK_1_PW2 (CFG_AW + CFG_WW + 5)
#define CFG_BLOCK_1_SW0 CFG_SW
#define CFG_BLOCK_1_SW1 CFG_SW
#define CFG_BLOCK_1_SW2 CFG_SW
#define CFG_BLOCK_1_BW0 CFG_BW
#define CFG_BLOCK_1_BW1 CFG_BW
#define CFG_BLOCK_1_BW2 CFG_BW

// BLOCK_2
#define CFG_BLOCK_2_PIC 6
#define CFG_BLOCK_2_PC 12
#define CFG_BLOCK_2_POC 8
#define CFG_BLOCK_2_IC 24
#define CFG_BLOCK_2_C 96
#define CFG_BLOCK_2_OC 32
#define CFG_BLOCK_2_H 16
#define CFG_BLOCK_2_W 16
#define CFG_BLOCK_2_PW0 (CFG_AW + CFG_WW + 5)
#define CFG_BLOCK_2_PW1 (CFG_AW + CFG_WW + 4)
#define CFG_BLOCK_2_PW2 (CFG_AW + CFG_WW + 5)
#define CFG_BLOCK_2_SW0 CFG_SW
#define CFG_BLOCK_2_SW1 CFG_SW
#define CFG_BLOCK_2_SW2 CFG_SW
#define CFG_BLOCK_2_BW0 CFG_BW
#define CFG_BLOCK_2_BW1 CFG_BW
#define CFG_BLOCK_2_BW2 CFG_BW

// BLOCK_3
#define CFG_BLOCK_3_PIC 8
#define CFG_BLOCK_3_PC 16
#define CFG_BLOCK_3_POC 8
#define CFG_BLOCK_3_IC 32
#define CFG_BLOCK_3_C 128
#define CFG_BLOCK_3_OC 32
#define CFG_BLOCK_3_H 8
#define CFG_BLOCK_3_W 8
#define CFG_BLOCK_3_PW0 (CFG_AW + CFG_WW + 5)
#define CFG_BLOCK_3_PW1 (CFG_AW + CFG_WW + 4)
#define CFG_BLOCK_3_PW2 (CFG_AW + CFG_WW + 5)
#define CFG_BLOCK_3_SW0 CFG_SW
#define CFG_BLOCK_3_SW1 CFG_SW
#define CFG_BLOCK_3_SW2 CFG_SW
#define CFG_BLOCK_3_BW0 CFG_BW
#define CFG_BLOCK_3_BW1 CFG_BW
#define CFG_BLOCK_3_BW2 CFG_BW
#define CFG_BLOCK_3_IW (CFG_SW + 1)

// BLOCK_4
#define CFG_BLOCK_4_PIC 8
#define CFG_BLOCK_4_PC 16
#define CFG_BLOCK_4_POC 16
#define CFG_BLOCK_4_IC 32
#define CFG_BLOCK_4_C 128
#define CFG_BLOCK_4_OC 64
#define CFG_BLOCK_4_H 8
#define CFG_BLOCK_4_W 8
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
#define CFG_BLOCK_5_PIC 16
#define CFG_BLOCK_5_PC 32
#define CFG_BLOCK_5_POC 16
#define CFG_BLOCK_5_IC 64
#define CFG_BLOCK_5_C 256
#define CFG_BLOCK_5_OC 64
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
#define CFG_BLOCK_6_PIC 16
#define CFG_BLOCK_6_PC 16
#define CFG_BLOCK_6_POC 8
#define CFG_BLOCK_6_IC 64
#define CFG_BLOCK_6_C 256
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

// CONV8
#define CFG_CONV8_PIC 8
#define CFG_CONV8_POC 24
#define CFG_CONV8_IC 72
#define CFG_CONV8_OC 96
#define CFG_CONV8_H 4
#define CFG_CONV8_W 4
#define CFG_CONV8_PW (CFG_AW + CFG_WW + 7)
#define CFG_CONV8_SW CFG_SW
#define CFG_CONV8_BW CFG_BW

// FC
#define CFG_FC_PIC 24
#define CFG_FC_POC 2
#define CFG_FC_IC 96
#define CFG_FC_OC 4
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
