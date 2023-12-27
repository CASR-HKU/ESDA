#include "fixgmp.h"

#define PRAGMA_SUB(x) _Pragma(#x)
#define DO_PRAGMA(x) PRAGMA_SUB(x)


#include "hls_stream.h"
#include "ap_int.h"
#include <iostream>
#include <string>
#include <fstream>

#include <boost/integer/static_log2.hpp>
#include <boost/integer/static_min_max.hpp>
#include <boost/integer/common_factor.hpp>

using namespace std;
#include "para.h"
#include "type.h"
#include "mem.h"
// #include "mask.h"
#include "linebuffer.h"
#include "conv.h"
#include "conv_pack.h"
#include "weight.h"

void top(ap_int<CFG_AW * CFG_TOP_PIC> *act_in,
         ap_int<32> *act_out, ap_int<CFG_MW> *mask,
         int num_nz);
