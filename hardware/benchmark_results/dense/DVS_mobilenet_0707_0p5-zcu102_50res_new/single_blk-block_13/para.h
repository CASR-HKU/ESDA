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

// BLOCK_13
#define CFG_BLOCK_13_PIC 6
#define CFG_BLOCK_13_PC 12
#define CFG_BLOCK_13_POC 8
#define CFG_BLOCK_13_IC 48
#define CFG_BLOCK_13_C 288
#define CFG_BLOCK_13_OC 80
#define CFG_BLOCK_13_H 8
#define CFG_BLOCK_13_W 8
#define CFG_BLOCK_13_PW0 (CFG_AW + CFG_WW + 6)
#define CFG_BLOCK_13_PW1 (CFG_AW + CFG_WW + 4)
#define CFG_BLOCK_13_PW2 (CFG_AW + CFG_WW + 7)
#define CFG_BLOCK_13_SW0 CFG_SW
#define CFG_BLOCK_13_SW1 CFG_SW
#define CFG_BLOCK_13_SW2 CFG_SW
#define CFG_BLOCK_13_BW0 CFG_BW
#define CFG_BLOCK_13_BW1 CFG_BW
#define CFG_BLOCK_13_BW2 CFG_BW

// TOP
#define CFG_TOP_IC CFG_BLOCK_13_IC
#define CFG_TOP_PIC CFG_BLOCK_13_PIC
#define CFG_TOP_IH CFG_BLOCK_13_H
#define CFG_TOP_IW CFG_BLOCK_13_W
#define CFG_TOP_OC CFG_BLOCK_13_OC
#define CFG_TOP_POC CFG_BLOCK_13_POC
#define CFG_TOP_OH (CFG_BLOCK_13_H/2)
#define CFG_TOP_OW (CFG_BLOCK_13_W/2)

#endif
