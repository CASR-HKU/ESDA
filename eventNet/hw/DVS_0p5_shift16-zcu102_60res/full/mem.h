template <int PI, class T>
T ceil_div(T A) {
    T d = A / PI;
    if (A > d * PI) {
        return d + 1;
    } else {
        return d;
    }
}

template <int PI, int IC, int AW, int MW, int HEIGHT, int WIDTH>
void M2S_mask_merge(ap_int<PI * AW> *act_in, ap_int<MW> *mask,
                    hls::stream<BundleT<PI, ap_int<AW> > > &act_out,
                    hls::stream<T_K> &token_out) {
#pragma HLS INLINE

    T_K token;
    int REP = ceil_div<MW>(HEIGHT * WIDTH);
    ap_int<MW> mask_buffer[HEIGHT * WIDTH / MW];
    ap_int<MW> mask_pack = 0;

    for (int rep = 0; rep < REP; rep++) {
#pragma HLS PIPELINE II = 1
        mask_buffer[rep] = mask[rep];
    }

    int count = 0;
    int x_index = 0;
    int y_index = 0;
    int index = 0;
    int ICPI = IC / PI;

    BundleT<PI, ap_int<AW> > pack;
#pragma HLS ARRAY_PARTITION variable = pack.data complete dim = 0

    for (int rep = 0; rep < REP; rep++) {
        mask_pack = mask_buffer[rep];
        for (int i = 0; i < MW; i++) {
#pragma HLS PIPELINE II = 1
            bool nz_flag = mask_pack[i];
            token.x = x_index;
            token.y = y_index;
            token.end = 0;
            if (nz_flag) {
                token_out.write(token);
                index = (x_index + y_index * WIDTH) * ICPI;
                for (int ic = 0; ic < ICPI; ic++) {
#pragma HLS pipeline II = 1
                    ap_int<PI *AW> m_read = act_in[index++];
                    for (ap_uint<7> pi = 0; pi < PI; pi++) {
                        pack.data[pi] =
                            m_read.range((pi + 1) * AW - 1, pi * AW);
                    }
                    act_out.write(pack);
                }
            }
            cout << count++ << " nz_flag:" << nz_flag << " x:" << token.x
                 << " y:" << token.y << endl;
            x_index++;
            if (x_index == WIDTH) {
                x_index = 0;
                y_index++;
            }
        }
    }

    token.x = 255;
    token.y = 255;
    token.end = 1;
    token_out.write(token);
}

// template <int MW, int HEIGHT, int WIDTH>
// void M2S_mask(ap_int<MW> *mask, hls::stream<T_K> &token_out,
//               hls::stream<ap_int<MW> > &mask_out) {
//     T_K token;
//     // ap_uint<16> REP = ceil_div<MW>(HEIGHT * WIDTH);

//     static const int WIDTH_DIV_ROUND = (WIDTH + MW - 1) / MW;

//     ap_int<MW> mask_buffer[HEIGHT * WIDTH_DIV_ROUND];

//     for (int rep = 0; rep < HEIGHT * WIDTH_DIV_ROUND; rep++) {
// #pragma HLS PIPELINE II = 1
//         ap_uint<MW> mask_read = mask[rep];
//         mask_buffer[rep] = mask_read;
//         mask_out.write(mask_read);
//     }

//     int count = 0;
//     int index = 0;
//     ap_uint<8> x_index = 0;
//     ap_uint<8> y_index = 0;

//     for (ap_uint<16> rep = 0; rep < HEIGHT * WIDTH_DIV_ROUND; rep++) {
//         ap_int<MW> mask_pack = mask_buffer[rep];
//         for (ap_uint<16> i = 0; i < MW; i++) {
// #pragma HLS PIPELINE II = 1
//             bool nz_flag = mask_pack[i];
//             token.x = x_index;
//             token.y = y_index;
//             token.end = 0;
//             if (nz_flag) {
//                 token_out.write(token);
//                 // cout<<count++<<" nz_flag:"<<nz_flag<<" x:"<<token.x<<"
//                 // y:"<<token.y<<endl;
//             }
//             x_index++;
//             if (x_index == WIDTH) {
//                 x_index = 0;
//                 y_index++;
//             }
//         }
//     }

//     // end token
//     token.x = 255;
//     token.y = 255;
//     token.end = 1;
//     token_out.write(token);
// }

template <int MW, int HEIGHT, int WIDTH>
void M2S_mask(ap_int<MW> *mask, hls::stream<T_K> &token_out,
              hls::stream<ap_int<MW> > &mask_out) {
    T_K token;
    // ap_uint<16> REP = ceil_div<MW>(HEIGHT * WIDTH);

    static const int WIDTH_DIV_ROUND = (WIDTH + MW - 1) / MW;
    static const int WIDTH_ROUND = WIDTH_DIV_ROUND * MW;

    ap_int<MW> mask_buffer[HEIGHT][WIDTH_DIV_ROUND];

    int read_count = 0;
    for (int h = 0; h < HEIGHT; h++) {
        for (int w = 0; w < WIDTH_DIV_ROUND; w++) {
#pragma HLS PIPELINE II = 1
            ap_uint<MW> mask_read = mask[read_count++];
            mask_buffer[h][w] = mask_read;
            mask_out.write(mask_read);
        }
    }

    int count = 0;
    int index = 0;
    ap_uint<8> x_index = 0;
    ap_uint<8> y_index = 0;

    int out_count = 0;
    for (ap_uint<16> h = 0; h < HEIGHT; h++) {
        for (ap_uint<16> w = 0; w <WIDTH_DIV_ROUND; w++) {
            ap_int<MW> mask_pack = mask_buffer[h][w];
            for (ap_uint<16> i = 0; i < MW; i++) {
#pragma HLS PIPELINE II = 1
                bool nz_flag = mask_pack[i];
                token.x = w * MW + i;
                token.y = h;
                token.end = 0;
                if (nz_flag) {
                    token_out.write(token);
                    // cout<<count++<<" nz_flag:"<<nz_flag<<" x:"<<token.x<<"
                    // y:"<<token.y<<endl;
                }
            }
        }
    }

    // end token
    token.x = 255;
    token.y = 255;
    token.end = 1;
    token_out.write(token);
}

template <int MW, int HEIGHT, int WIDTH>
void mask_stride2(hls::stream<ap_int<MW> > &mask_in,
                  hls::stream<T_K> &token_out) {
    
    static const int WIDTH_ROUND = ((WIDTH + MW - 1) / MW) * MW;
    static const int WIDTH_DIV_ROUND = (WIDTH + MW - 1) / MW;
    
    T_K token;
    ap_uint<16> REP = HEIGHT;
    ap_int<WIDTH_ROUND> mask_buffer[HEIGHT];
#pragma HLS ARRAY_PARTITION variable = mask_buffer cyclic factor = 2 dim = 1

    for(int i = 0; i < HEIGHT; i++){
        mask_buffer[i] = 0;
    }

    for (int rep = 0; rep < HEIGHT; rep++) {
        for(int i = 0; i < WIDTH_DIV_ROUND; i++){
#pragma HLS PIPELINE II = 1
            ap_int<MW> mask_pack = mask_in.read();
            mask_buffer[rep].range((i + 1) * MW - 1, i * MW) = mask_pack;
        }
    }

    int count = 0;
    int index = 0;
    ap_uint<8> x_index = 0;
    ap_uint<8> y_index = 0;

    ap_int<WIDTH_ROUND> mask_pack_even, mask_pack_odd, mask_or, mask_or_odd,
        mask_or_even;
    for (ap_uint<16> y = 0; y < HEIGHT / 2; y++) {
        mask_pack_even = mask_buffer[y * 2];
        mask_pack_odd = mask_buffer[y * 2 + 1];
        mask_or = mask_pack_even | mask_pack_odd;
        for (ap_uint<16> x = 0; x < WIDTH_ROUND / 2; x++) {
#pragma HLS PIPELINE II = 1
            bool nz_flag = mask_or[x * 2] | mask_or[x * 2 + 1];
            if (nz_flag) {
                token.x = x * 2;
                token.y = y * 2;
                token.end = 0;
                token_out.write(token);
                // cout<<count++<<" nz_flag:"<<nz_flag<<" x:"<<token.x<<"
                // y:"<<token.y<<endl;
            }
        }
    }

    // end token
    token.x = 255;
    token.y = 255;
    token.end = 1;
    token_out.write(token);
}

template <int PI, int AW, int IC>
void read_sparse_input(ap_int<PI * AW> *act_in,
                       hls::stream<BundleT<PI, ap_int<AW> > > &act_out,
                       int num_nz) {
    BundleT<PI, ap_int<AW> > out_pack;
#pragma HLS ARRAY_PARTITION variable = out_pack.data complete dim = 0

    for (int i = 0; i < num_nz * IC / PI; i++) {
#pragma HLS PIPELINE II = 1
        ap_int<PI *AW> tmp = act_in[i];
        for (int j = 0; j < PI; j++) {
#pragma HLS UNROLL
            out_pack.data[j] = tmp.range((j + 1) * AW - 1, j * AW);
        }
        act_out.write(out_pack);
    }
}

template <int MW, int HEIGHT, int WIDTH>
void read_mask_to_token(ap_uint<MW> *mask, hls::stream<T_K> &token_out) {
    T_K key;
    int REP = ceil_div<MW>(HEIGHT * WIDTH);
    ap_uint<MW> mask_pack = 0;
    int read_count = 0;

    ap_uint<MW> mask_buffer[HEIGHT * WIDTH / MW];

    for (int rep = 0; rep < REP; rep++) {
        mask_pack = mask[rep];
        mask_buffer[rep] = mask_pack;
    }

    ap_uint<16> index = 0;
    for (int h = 0; h < HEIGHT; h++) {
        for (int w = 0; w < WIDTH; w++) {
#pragma HLS pipeline II = 1
            int div = index.range(15, 8);
            int mod = index.range(MW, 0);
            ap_uint<MW> mask_pack = mask_buffer[div];
            index++;
            // cout<<count++<<" mask:"<<mask_pack[mod]<<endl;
            if (mask_pack[mod] == 1) {
                key.x = w;
                key.y = h;
                key.end = 0;
                token_out.write(key);
                // cout<<"key.x:"<<key.x<<" key.y:"<<key.y<<endl;
            }
        }
    }

    key.end = 1;
    token_out.write(key);
}

// read weights
template <int PI, int IC, int WW>
void read_weights_3x3_dw(ap_int<PI * WW> *weight_in,
                         ap_int<PI * WW> weight[9][IC / PI]) {
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < IC / PI; j++) {
#pragma HLS PIPELINE II = 1
            ap_int<PI *WW> tmp = weight_in[i * IC / PI + j];
            weight[i][j] = tmp;
        }
    }
}

template <int PI, int IC, int OC, int WW>
void read_weights_1x1(ap_int<PI * WW> *weight_in,
                      ap_int<PI * WW> weight[OC][IC / PI]) {
    cout << "read_weights_1x1" << endl;
    for (int i = 0; i < OC; i++) {
        for (int j = 0; j < IC / PI; j++) {
#pragma HLS PIPELINE II = 1
            ap_int<PI *WW> tmp = weight_in[i * IC / PI + j];
            weight[i][j] = tmp;
            cout << "weight[" << i << "][" << j << "]:" << weight[i][j] << "\t";
        }
        cout << endl;
    }
}

// write output
template <int PO, int OC, int AW, int HEIGHT, int WIDTH>
void write_output(hls::stream<BundleT<PO, ap_int<AW> > > &out_s,
                  hls::stream<T_K> &token_out, ap_int<PO * AW> *out) {
    int count = 0;
    int index = 0;
    for (int rep = 0; rep < HEIGHT * WIDTH + 1; rep++) {
        T_K token = token_out.read();
        if (token.end == 1) break;
        // int index = (token.x + token.y * WIDTH) * OC / PO;

        for (int i = 0; i < OC / PO; i++) {
#pragma HLS PIPELINE II = 1
            BundleT<PO, ap_int<AW> > tmp = out_s.read();
            ap_int<PO *AW> pack = 0;
            for (int j = 0; j < PO; j++) {
#pragma HLS UNROLL
                pack.range((j + 1) * AW - 1, j * AW) = tmp.data[j];
            }

            out[index++] = pack;
        }
    }
}

// duplicate stream
// template<int PI, int IC, int AW, int HEIGHT, int WIDTH>
// void duplicate_stream(
// 	hls::stream<BundleT<PI, ap_int<AW> > > &act_in,
// 	hls::stream<BundleT<PI, ap_int<AW> > > &act_out_0,
// 	hls::stream<BundleT<PI, ap_int<AW> > > &act_out_1,
// 	hls::stream<T_K> &token_in,
// 	hls::stream<T_K> &token_out
// ){

// 	T_K token;
// 	BundleT<PI, ap_int<AW> > in_read;

// 	for(int rep = 0; rep < HEIGHT * WIDTH + 1; rep++){
// 		token = token_in.read();
// 		token_out.write(token);
// 		if (token.end == 1) break;
// 		for(int i = 0; i < IC / PI; i++){
// #pragma HLS PIPELINE II=1
// 			in_read = act_in.read();
// 			act_out_0.write(in_read);
// 			act_out_1.write(in_read);
// 		}
// 	}
// }

// // duplicate stream
// template<int PI, int PO, int IC, int AW, int HEIGHT, int WIDTH>
// void duplicate_stream(
// 	hls::stream<BundleT<PI, ap_int<AW> > > &act_in,
// 	hls::stream<BundleT<PI, ap_int<AW> > > &act_out,
// 	hls::stream<BundleT<PO, ap_int<AW> > > &act_id,
// 	hls::stream<T_K> &token_in,
// 	hls::stream<T_K> &token_out
// ){

// 	T_K token;
// 	BundleT<PI, ap_int<AW> > in_pack;
// 	BundleT<PO, ap_int<AW> > id_pack;

// 	cout<<"PI:"<<PI<<" PO:"<<PO<<" IC:"<<IC<<endl;

// 	for(int rep = 0; rep < HEIGHT * WIDTH + 2; rep++){
// 		token = token_in.read();
// 		token_out.write(token);
// 		if (token.end == 1) break;
// 		if(PI > PO){
// 			for(int i = 0; i < IC / PI; i++){
// #pragma HLS PIPELINE
// 				in_pack = act_in.read();
// 				act_out.write(in_pack);
// 				for(int j = 0; j < PI / PO; j++){
// 					for(int po = 0; po < PO; po++){
// 						id_pack.data[po] =
// in_pack.data[po
// +
// j
// * PO];
// 					}
// 					act_id.write(id_pack);
// 				}
// 			}
// 		}
// 		else if(PI < PO){
// 			for(int i = 0; i < IC / PO; i++){
// 				for(int j = 0; j < PO / PI; j++){
// #pragma HLS PIPELINE
// 					in_pack = act_in.read();
// 					act_out.write(in_pack);
// 					for(int pi = 0; pi < PI; pi++){
// 						id_pack.data[pi + j * PI] =
// in_pack.data[pi];
// 					}
// 				}
// 				act_id.write(id_pack);
// 			}
// 		}
// 		else{
// 			for(int i = 0; i < IC / PI; i++){
// #pragma HLS PIPELINE
// 				in_pack = act_in.read();
// 				for(int j = 0; j < PI; j++){
// 					id_pack.data[j] = in_pack.data[j];
// 				}
// 				act_out.write(in_pack);
// 				act_id.write(id_pack);
// 			}
// 		}
// 	}
// }

// duplicate stream
template <int PI, int PO, int IC, int AW, int HEIGHT, int WIDTH>
void duplicate_stream(hls::stream<BundleT<PI, ap_int<AW> > > &act_in,
                      hls::stream<BundleT<PI, ap_int<AW> > > &act_out,
                      hls::stream<BundleT<PO, ap_int<AW> > > &act_id,
                      hls::stream<T_K> &token_in, hls::stream<T_K> &token_out) {
    static const int LCM = boost::integer::static_lcm<PI, PO>::value;

    T_K token;
    BundleT<PI, ap_int<AW> > in_pack;
    BundleT<PO, ap_int<AW> > id_pack;
    BundleT<LCM, ap_int<AW> > lcm_pack;

    cout << "PI:" << PI << " PO:" << PO << " IC:" << IC << endl;

    for (int rep = 0; rep < HEIGHT * WIDTH + 2; rep++) {
        token = token_in.read();
        token_out.write(token);
        if (token.end == 1) break;
        if (PI == PO) {
            for (int i = 0; i < IC / PI; i++) {
#pragma HLS PIPELINE
                in_pack = act_in.read();
                for (int j = 0; j < PI; j++) {
                    id_pack.data[j] = in_pack.data[j];
                }
                act_out.write(in_pack);
                act_id.write(id_pack);
            }
        } else {
            for (int i = 0; i < IC / LCM; i++) {
                for (int j = 0; j < LCM / PI; j++) {
#pragma HLS PIPELINE
                    in_pack = act_in.read();
                    act_out.write(in_pack);
                    for (int pi = 0; pi < PI; pi++) {
                        lcm_pack.data[pi + j * PI] = in_pack.data[pi];
                    }
                }
                for (int k = 0; k < LCM / PO; k++) {
#pragma HLS PIPELINE
                    for (int po = 0; po < PO; po++) {
                        id_pack.data[po] = lcm_pack.data[po + k * PO];
                    }
                    act_id.write(id_pack);
                }
            }
        }
    }
}

template <int MW, int HEIGHT, int WIDTH>
void M2S_mask_first_layer(ap_uint<WIDTH> *mask, hls::stream<T_K> &token_out,
                          hls::stream<ap_uint<WIDTH> > &mask_out) {
    T_K token;
    ap_uint<16> REP = ceil_div<MW>(HEIGHT * WIDTH);
    ap_int<MW> mask_buffer[HEIGHT * WIDTH / MW];
    ap_int<MW> mask_pack = 0;

    for (int rep = 0; rep < REP; rep++) {
#pragma HLS PIPELINE II = 1
        ap_uint<WIDTH> mask_pack = mask[rep];
        mask_buffer[rep] = mask_pack;
        mask_out.write(mask_pack);
    }

    int count = 0;
    int index = 0;
    ap_uint<8> x_index = 0;
    ap_uint<8> y_index = 0;

    for (ap_uint<16> rep = 0; rep < REP; rep++) {
        mask_pack = mask_buffer[rep];
        for (ap_uint<16> i = 0; i < MW; i++) {
#pragma HLS PIPELINE II = 1
            bool nz_flag = mask_pack[i];
            token.x = x_index;
            token.y = y_index;
            token.end = 0;
            if (nz_flag) {
                token_out.write(token);
                // cout<<count++<<" nz_flag:"<<nz_flag<<" x:"<<token.x<<"
                // y:"<<token.y<<endl;
            }
            x_index++;
            if (x_index == WIDTH) {
                x_index = 0;
                y_index++;
            }
        }
    }

    // end token
    token.x = 255;
    token.y = 255;
    token.end = 1;
    token_out.write(token);
}

// //adjust stream
// template<int PI, int PF, int AW, int HEIGHT, int WIDTH>
// void adjust_stream(
// 	hls::stream<BundleT<PI, ap_int<AW> > > &act_in,
// 	hls::stream<BundleT<PF, ap_int<AW> > > &act_out,
// 	hls::stream<T_K> &token_in,
// 	hls::stream<T_K> &token_out
// ){
// 	T_K token;
// 	BundleT<PI, ap_int<AW> > in_data;
// 	BundleT<PF, ap_int<AW> > out_data;

// 	for(int rep = 0; rep < HEIGHT * WIDTH + 1; rep++){
// 		token = token_in.read();
// 		if (token.end == 1) break;
// 		if(PI > PF){
// 			for(int i = 0; i < PI; i++){
// 				in_data = act_in.read();
// 				for(int j = 0; j < PI / PF; j++){

// 			}
// 		}

// )