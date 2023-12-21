#ifndef __PARA_H__
#define __PARA_H__

// COMMON
#define CFG_AW 8   // AW
#define CFG_WW 8   // WW
#define CFG_PW 32  // PSUMW
#define CFG_SW 16  // SCALEW
#define CFG_BW 16  // BIASW
#define CFG_TW 8   // TOKENW
#define CFG_MW 16  // to be very carefull
#define CFG_EXP 16 // EXP

// BLOCK_1
#define CFG_BLOCK_1_PIC 4
#define CFG_BLOCK_1_PC 2
#define CFG_BLOCK_1_POC 4
#define CFG_BLOCK_1_IC 8
#define CFG_BLOCK_1_C 48
#define CFG_BLOCK_1_OC 12
#define CFG_BLOCK_1_H 64
#define CFG_BLOCK_1_W 64
#define CFG_BLOCK_1_PW0 (CFG_AW + CFG_WW + 3)
#define CFG_BLOCK_1_PW1 (CFG_AW + CFG_WW + 4)
#define CFG_BLOCK_1_PW2 (CFG_AW + CFG_WW + 4)
#define CFG_BLOCK_1_SW0 CFG_SW
#define CFG_BLOCK_1_SW1 CFG_SW
#define CFG_BLOCK_1_SW2 CFG_SW
#define CFG_BLOCK_1_BW0 CFG_BW
#define CFG_BLOCK_1_BW1 CFG_BW
#define CFG_BLOCK_1_BW2 CFG_BW

// TOP
#define CFG_TOP_IC CFG_BLOCK_1_IC
#define CFG_TOP_PIC CFG_BLOCK_1_PIC
#define CFG_TOP_IH CFG_BLOCK_1_H
#define CFG_TOP_IW CFG_BLOCK_1_W
#define CFG_TOP_OC CFG_BLOCK_1_OC
#define CFG_TOP_POC CFG_BLOCK_1_POC
#define CFG_TOP_OH (CFG_BLOCK_1_H/2)
#define CFG_TOP_OW (CFG_BLOCK_1_W/2)

#endif
