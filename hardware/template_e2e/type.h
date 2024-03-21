
typedef struct T_K{
	ap_uint<1> end;
	ap_uint<8> x;
	ap_uint<8> y;
} T_K;

template <unsigned int N, typename T>
struct BundleT {
	T data[N];
};

union Float_Int_T {
    float as_float;
    int as_int;

    Float_Int_T(){
        this->as_int = 0;
    }
};

typedef ap_uint<4> T_OFFSET;
#define end_3x3 15