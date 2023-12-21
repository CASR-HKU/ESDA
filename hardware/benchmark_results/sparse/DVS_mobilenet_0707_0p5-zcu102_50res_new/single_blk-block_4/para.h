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

// BLOCK_4
#define CFG_BLOCK_4_PIC 4
#define CFG_BLOCK_4_PC 8
#define CFG_BLOCK_4_POC 4
#define CFG_BLOCK_4_IC 16
#define CFG_BLOCK_4_C 96
#define CFG_BLOCK_4_OC 16
#define CFG_BLOCK_4_H 16
#define CFG_BLOCK_4_W 16
#define CFG_BLOCK_4_PW0 (CFG_AW + CFG_WW + 4)
#define CFG_BLOCK_4_PW1 (CFG_AW + CFG_WW + 4)
#define CFG_BLOCK_4_PW2 (CFG_AW + CFG_WW + 4)
#define CFG_BLOCK_4_SW0 CFG_SW
#define CFG_BLOCK_4_SW1 CFG_SW
#define CFG_BLOCK_4_SW2 CFG_SW
#define CFG_BLOCK_4_BW0 CFG_BW
#define CFG_BLOCK_4_BW1 CFG_BW
#define CFG_BLOCK_4_BW2 CFG_BW
#define CFG_BLOCK_4_IW (CFG_SW + 1)

// TOP
#define CFG_TOP_IC CFG_BLOCK_4_IC
#define CFG_TOP_PIC CFG_BLOCK_4_PIC
#define CFG_TOP_IH CFG_BLOCK_4_H
#define CFG_TOP_IW CFG_BLOCK_4_W
#define CFG_TOP_OC CFG_BLOCK_4_OC
#define CFG_TOP_POC CFG_BLOCK_4_POC
#define CFG_TOP_OH (CFG_BLOCK_4_H/1)
#define CFG_TOP_OW (CFG_BLOCK_4_W/1)

#endif
