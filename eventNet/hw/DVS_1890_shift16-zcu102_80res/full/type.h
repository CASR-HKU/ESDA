
typedef struct T_K{
	ap_uint<1> end;
	ap_uint<8> x;
	ap_uint<8> y;
} T_K;

template <unsigned int N, typename T>
struct BundleT {
	T data[N];
};



typedef ap_uint<4> T_OFFSET;
#define end_3x3 15