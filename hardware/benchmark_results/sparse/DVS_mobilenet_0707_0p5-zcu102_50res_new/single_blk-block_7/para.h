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

// BLOCK_7
#define CFG_BLOCK_7_PIC 4
#define CFG_BLOCK_7_PC 8
#define CFG_BLOCK_7_POC 4
#define CFG_BLOCK_7_IC 32
#define CFG_BLOCK_7_C 192
#define CFG_BLOCK_7_OC 32
#define CFG_BLOCK_7_H 8
#define CFG_BLOCK_7_W 8
#define CFG_BLOCK_7_PW0 (CFG_AW + CFG_WW + 5)
#define CFG_BLOCK_7_PW1 (CFG_AW + CFG_WW + 4)
#define CFG_BLOCK_7_PW2 (CFG_AW + CFG_WW + 5)
#define CFG_BLOCK_7_SW0 CFG_SW
#define CFG_BLOCK_7_SW1 CFG_SW
#define CFG_BLOCK_7_SW2 CFG_SW
#define CFG_BLOCK_7_BW0 CFG_BW
#define CFG_BLOCK_7_BW1 CFG_BW
#define CFG_BLOCK_7_BW2 CFG_BW
#define CFG_BLOCK_7_IW (CFG_SW + 1)

// TOP
#define CFG_TOP_IC CFG_BLOCK_7_IC
#define CFG_TOP_PIC CFG_BLOCK_7_PIC
#define CFG_TOP_IH CFG_BLOCK_7_H
#define CFG_TOP_IW CFG_BLOCK_7_W
#define CFG_TOP_OC CFG_BLOCK_7_OC
#define CFG_TOP_POC CFG_BLOCK_7_POC
#define CFG_TOP_OH (CFG_BLOCK_7_H/1)
#define CFG_TOP_OW (CFG_BLOCK_7_W/1)

#endif
