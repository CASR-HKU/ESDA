template <int _W_1, int _W_2>
ap_int<_W_1 + _W_2> DSP_AM(ap_int<_W_1> in1, ap_int<_W_1> in2,
                           ap_int<_W_2> in3) {
#pragma HLS INLINE OFF
    ap_int<_W_1> add_temp = in1 + in2;
    ap_int<_W_1 + _W_2> mul_temp = add_temp * in3;
    return mul_temp;
}

template <int PI, int IC, int HEIGHT, int WIDTH, int AW, int WW, int PSUMW>
void conv_3x3_dw_kernel(hls::stream<BundleT<9, ap_int<PI * AW>>> &act_in,
                        hls::stream<BundleT<PI, ap_int<PSUMW>>> &psum_out,
                        hls::stream<T_K> &token_in, hls::stream<T_K> &token_out,
                        const ap_int<PI * WW> w_buffer[9][IC / PI]) {
    static const int C_W = boost::static_log2<IC>::value + 2;
    static const int HW_W = boost::static_log2<HEIGHT * WIDTH>::value + 2;

    typedef ap_uint<C_W> T_C;
#pragma HLS bind_storage variable = w_buffer type = rom_2p impl = BRAM

    BundleT<9, ap_int<PI * WW>> weight_pack;
#pragma HLS ARRAY_PARTITION variable = weight_pack.data complete dim = 0
    BundleT<9, ap_int<PI * AW>> act_window;
#pragma HLS ARRAY_PARTITION variable = act_window.data complete dim = 0
    BundleT<PI, ap_int<PSUMW>> psum_pack;
#pragma HLS ARRAY_PARTITION variable = psum_pack.data complete dim = 0

    for (ap_uint<HW_W> rep = 0; rep < HEIGHT * WIDTH + 2; rep++) {
        T_K token = token_in.read();
        token_out.write(token);
        if (token.end == 1) break;

        for (T_C ic = 0; ic < IC / PI; ic++) {
#pragma HLS PIPELINE II = 1
            act_window = act_in.read();
            for (ap_uint<4> k = 0; k < 9; k++) {
#pragma HLS UNROLL
                weight_pack.data[k] = w_buffer[k][ic];
            }
            for (T_C pi = 0; pi < PI; pi++) {
#pragma HLS UNROLL
                ap_int<PSUMW> psum = 0;
                for (ap_uint<4> k = 0; k < 9; k++) {
#pragma HLS UNROLL
                    ap_int<AW> activation =
                        (ap_int<AW>)act_window.data[k].range((pi + 1) * AW - 1,
                                                             pi * AW);
                    ap_int<WW> weight = (ap_int<WW>)weight_pack.data[k].range(
                        (pi + 1) * WW - 1, pi * WW);
                    psum += activation * weight;
                    // cout << " a:" << activation << " w:" << weight
                    //      << " p:" << psum;
                }
                // cout << endl;
                psum_pack.data[pi] = psum;
            }
            psum_out.write(psum_pack);
        }
    }
}

template <int PI, int IC, int HEIGHT, int WIDTH, int PSUMW, int AW, int SCALEW,
          int BIASW, int EXP, bool relu>
void quantize(hls::stream<BundleT<PI, ap_int<PSUMW>>> &psum_in,
              hls::stream<BundleT<PI, ap_int<AW>>> &act_out,
              hls::stream<T_K> &token_in, hls::stream<T_K> &token_out,
              const ap_int<SCALEW + BIASW> scale_buffer[IC]) {
    static const int C_W = boost::static_log2<IC>::value + 2;
    static const int HW_W = boost::static_log2<HEIGHT * WIDTH>::value + 2;
    static const ap_uint<EXP> round_shift = 1 << (EXP - 1);

#pragma HLS bind_storage variable = scale_buffer type = rom_2p impl = BRAM

    DO_PRAGMA(HLS ARRAY_PARTITION variable = scale_buffer cyclic factor =
                  (PI / 2) dim = 1)

    BundleT<PI, ap_int<PSUMW>> psum_pack;
#pragma HLS ARRAY_PARTITION variable = psum_pack.data complete dim = 0
    BundleT<PI, ap_int<AW>> act_pack;
#pragma HLS ARRAY_PARTITION variable = act_pack.data complete dim = 0

    ap_int<AW> quantized_act;
    const int high = (1 << (AW - 1)) - 1;
    const int low = -(1 << (AW - 1)) + 1;
    int count = 0;

    for (ap_int<HW_W> r = 0; r < HEIGHT * WIDTH + 2; r++) {
        T_K token = token_in.read();
        token_out.write(token);
        // cout<<"token.x:"<<token.x<<"
        // token.y:"<<token.y<<"token.end:"<<token.end<<endl;
        if (token.end == 1) break;

        for (ap_uint<C_W> c = 0; c < IC / PI; c++) {
#pragma HLS PIPELINE II = 1
            psum_pack = psum_in.read();
            for (ap_uint<C_W> p = 0; p < PI; p++) {
#pragma HLS UNROLL
                ap_int<PSUMW> psum = psum_pack.data[p];
                ap_int<SCALEW + BIASW> scale_bias = scale_buffer[c * PI + p];
                ap_uint<SCALEW> scale =
                    (ap_uint<SCALEW>)scale_bias.range(SCALEW + BIASW - 1, BIASW);
                ap_int<BIASW> bias =
                    (ap_int<BIASW>)scale_bias.range(BIASW - 1, 0);

                ap_int<PSUMW + SCALEW + 1> psum_mul =
                    (psum + bias) * scale + round_shift;

                cout<<"psum:"<<psum<<" scale:"<<scale<<" bias:"<<bias<<" psum_mul:"<<psum_mul<<" round shift:"<<round_shift<<" exp:"<<EXP;

                psum_mul = psum_mul >> EXP;

                if (psum_mul > high) psum_mul = high;
                if (psum_mul < low) psum_mul = low;
                quantized_act = (ap_int<AW>)psum_mul;

                if (relu) {
                    act_pack.data[p] =
                        quantized_act < 0 ? (ap_int<AW>)0 : quantized_act;
                } else {
                    act_pack.data[p] = quantized_act;
                }
                cout<<" output:"<<act_pack.data[p]<<endl;
            }
            act_out.write(act_pack);
        }
    }
}

template <int PI, int IC, int HEIGHT, int WIDTH, int PSUMW, int AW, int SCALEW,
          int BIASW, int SCALEIDW, int EXP>
void quantize_id_add(hls::stream<BundleT<PI, ap_int<PSUMW>>> &psum_in,
                     hls::stream<BundleT<PI, ap_int<AW>>> &act_id,
                     hls::stream<BundleT<PI, ap_int<AW>>> &act_out,
                     hls::stream<T_K> &token_in, hls::stream<T_K> &token_out,
                     const ap_int<SCALEW + BIASW> scale_buffer[IC],
                     const ap_uint<SCALEIDW> id_scale) {
    static const int C_W = boost::static_log2<IC>::value + 2;
    static const int HW_W = boost::static_log2<HEIGHT * WIDTH>::value + 2;

#pragma HLS bind_storage variable = scale_buffer type = rom_2p impl = BRAM

    DO_PRAGMA(HLS ARRAY_PARTITION variable = scale_buffer cyclic factor =
                  (PI / 2) dim = 1)

    BundleT<PI, ap_int<PSUMW>> psum_pack;
#pragma HLS ARRAY_PARTITION variable = psum_pack.data complete dim = 0
    BundleT<PI, ap_int<AW>> act_pack;
#pragma HLS ARRAY_PARTITION variable = act_pack.data complete dim = 0

    BundleT<PI, ap_int<AW>> id_pack;
#pragma HLS ARRAY_PARTITION variable = id_pack.data complete dim = 0

    ap_int<AW> quantized_act;
    const int high = (1 << (AW - 1)) - 1;
    const int low = -(1 << (AW - 1)) + 1;

    const int high_1 = (1 << (AW)) - 1;
    const int low_1 = -(1 << (AW));

    int count = 0;

    const ap_uint<EXP> round_shift = 1 << (EXP - 1);

    ap_int<AW + 1> quantize_psum;
    ap_int<AW + 1> quantize_id;
    ap_int<AW + 2> final_sum;

    for (ap_int<HW_W> r = 0; r < HEIGHT * WIDTH + 2; r++) {
        T_K token = token_in.read();
        token_out.write(token);
        if (token.end == 1) break;

        for (ap_uint<C_W> c = 0; c < IC / PI; c++) {
#pragma HLS PIPELINE II = 1
            psum_pack = psum_in.read();
            id_pack = act_id.read();
            for (ap_uint<C_W> p = 0; p < PI; p++) {
#pragma HLS UNROLL
                ap_int<PSUMW> psum = psum_pack.data[p];
                ap_int<SCALEW + BIASW> scale_bias = scale_buffer[c * PI + p];
                ap_uint<SCALEW> scale = (ap_uint<SCALEW>)scale_bias.range(
                    SCALEW + BIASW - 1, BIASW);
                ap_int<BIASW> bias =
                    (ap_int<BIASW>)scale_bias.range(BIASW - 1, 0);

                ap_int<AW> id = id_pack.data[p];

                ap_int<PSUMW + SCALEW + 1> psum_mul =
                    ((psum + bias) * scale + round_shift);

                // cout<<count++;
                // cout<<" psum:"<<psum<<" scale:"<<scale<<" bias:"<<bias<<"
                // psum_mul:"<<psum_mul<<" round shift:"<<round_shift;

                psum_mul = psum_mul >> EXP;

                if (psum_mul > high_1) psum_mul = high_1;
                if (psum_mul < low_1) psum_mul = low_1;
                quantize_psum = psum_mul;

                ap_int<AW + SCALEIDW + 1> id_mul =
                    (id * id_scale + round_shift);
                // cout<<" SCALEWIDW:"<<SCALEIDW <<" id:"<<id<<"
                // id_scale:"<<id_scale<<" id_mul:"<<id_mul<<" round
                // shift:"<<round_shift;

                id_mul = id_mul >> EXP;

                // if (id_mul > high_1) id_mul = high_1;
                // if (id_mul < low_1) id_mul = low_1;
                quantize_id = id_mul;

                final_sum = quantize_psum + quantize_id;
                // cout<<" quantize_psum:"<<quantize_psum<<"
                // quantize_id:"<<quantize_id<<" final_sum:"<<final_sum<<endl;

                if (final_sum > high) final_sum = high;
                if (final_sum < low) final_sum = low;
                quantized_act = (ap_int<AW>)final_sum;
                act_pack.data[p] = quantized_act;
            }
            act_out.write(act_pack);
        }
    }
}

template <int PI, int PO, int IC, int OC, int HEIGHT, int WIDTH, int AW, int WW,
          int PSUMW>
void conv_1x1_kernel_dsp(hls::stream<BundleT<PI, ap_int<AW>>> &act_in,
                         hls::stream<BundleT<PO, ap_int<PSUMW>>> &psum_out,
                         hls::stream<T_K> &token_in,
                         hls::stream<T_K> &token_out,
                         const ap_int<PI * WW> w_buffer[OC][IC / PI]) {
    // const int C_W = 10;
    // const int HW_W = 16;
#pragma HLS bind_storage variable = w_buffer type = rom_2p impl = BRAM

    DO_PRAGMA(HLS ARRAY_PARTITION variable = w_buffer cyclic factor =
                  (PO / 2) dim = 1)

    static const int MAXC = boost::static_signed_max<IC, OC>::value;
    static const int C_W = boost::static_log2<MAXC>::value + 2;
    static const int HW_W = boost::static_log2<HEIGHT * WIDTH>::value + 2;

    BundleT<PO, ap_int<PI * WW>> weight_pack;
#pragma HLS ARRAY_PARTITION variable = weight_pack.data complete dim = 0

    BundleT<PI, ap_int<AW>> act_in_pack;
#pragma HLS ARRAY_PARTITION variable = act_in_pack.data complete dim = 0

    BundleT<PO, ap_int<PSUMW>> psum_pack;
#pragma HLS ARRAY_PARTITION variable = psum_pack.data complete dim = 0

    typedef ap_int<AW> T_ACT;
    typedef ap_int<WW> T_WEIGHT;
    typedef ap_int<PSUMW> T_PSUM;
    typedef ap_uint<C_W> T_C;

    ap_int<PI * AW> act_buffer[IC / PI];

    for (T_C oc = 0; oc < PO; oc++) {
#pragma HLS UNROLL
        psum_pack.data[oc] = 0;
    }

    for (ap_uint<HW_W> r = 0; r < HEIGHT * WIDTH + 2; r++) {
        T_K token = token_in.read();
        token_out.write(token);
        if (token.end == 1) break;

        for (T_C c = 0; c < IC / PI; c++) {
#pragma HLS PIPELINE II = 1
            act_in_pack = act_in.read();
            ap_int<PI *AW> act_tmp = 0;
            for (T_C p = 0; p < PI; p++) {
#pragma HLS UNROLL
                act_tmp.range((p + 1) * AW - 1, p * AW) = act_in_pack.data[p];
            }
            act_buffer[c] = act_tmp;
        }

        for (T_C oc = 0; oc < OC / PO; oc++) {
            for (T_C ic = 0; ic < IC / PI; ic++) {
#pragma HLS PIPELINE
                ap_int<PI *AW> act_c_pack = act_buffer[ic];
                for (T_C po = 0; po < PO; po++) {
                    weight_pack.data[po] = w_buffer[oc * PO + po][ic];
                }

                // pack 2 MAC on 1 DSP
                for (T_C po = 0; po < PO / 2; po++) {
#pragma HLS UNROLl
                    for (T_C pi = 0; pi < PI; pi++) {
#pragma HLS UNROLl
                        T_ACT activation =
                            (T_ACT)act_c_pack.range(AW * (pi + 1) - 1, AW * pi);
                        ap_int<18> in_expend = (ap_int<18>)activation;

                        T_WEIGHT w_0 =
                            (T_WEIGHT)(weight_pack.data[po * 2].range(
                                (pi + 1) * WW - 1, pi * WW));
                        T_WEIGHT w_1 =
                            (T_WEIGHT)(weight_pack.data[po * 2 + 1].range(
                                (pi + 1) * WW - 1, pi * WW));

                        ap_int<27> w_1_shift = 0;
                        ap_int<27> w_0_expend = (ap_int<27>)w_0;
                        w_1_shift.range(26, 18) = (ap_int<9>)w_1;

                        ap_int<48> mul_temp =
                            DSP_AM(w_1_shift, w_0_expend, in_expend);
                        ap_int<AW + WW> low =
                            (ap_int<AW + WW>)(mul_temp.range(AW + WW - 1, 0));
                        ap_int<AW + WW> high =
                            (ap_int<AW + WW>)(mul_temp.range(AW + WW - 1 + 18,
                                                             18)) +
                            mul_temp.range(AW + WW - 1, AW + WW - 1);
                        psum_pack.data[po * 2] += low;
                        psum_pack.data[po * 2 + 1] += high;
                    }
                }
            }
            psum_out.write(psum_pack);
            for (T_C p = 0; p < PO; p++) {
#pragma HLS UNROLL
                psum_pack.data[p] = 0;
            }
        }
    }
}

template <int PI, int IC, int HEIGHT, int WIDTH, int AW, int WW, int PSUMW>
void conv_3x3_dw_kernel_serial(
    hls::stream<BundleT<PI, ap_int<AW>>> &act_in,
    hls::stream<BundleT<PI, ap_int<PSUMW>>> &psum_out,
    hls::stream<T_K> &token_in, hls::stream<T_K> &token_out,
    hls::stream<T_OFFSET> &offset_s,
    const ap_int<PI * WW> w_buffer[9][IC / PI]) {
#pragma HLS bind_storage variable = w_buffer type = rom_2p impl = BRAM

    static const int C_W = boost::static_log2<IC>::value + 2;
    static const int HW_W = boost::static_log2<HEIGHT * WIDTH>::value + 2;

    typedef ap_uint<C_W> T_C;

    BundleT<9, ap_int<PI * WW>> weight_pack;
#pragma HLS ARRAY_PARTITION variable = weight_pack.data complete dim = 0
    BundleT<PI, ap_int<AW>> act_window;
#pragma HLS ARRAY_PARTITION variable = act_window.data complete dim = 0
    BundleT<PI, ap_int<PSUMW>> psum_pack;
#pragma HLS ARRAY_PARTITION variable = psum_pack.data complete dim = 0

    ap_int<PSUMW> psum_buffer[IC];
    DO_PRAGMA(HLS ARRAY_PARTITION variable = psum_buffer cyclic factor =
                  PI dim = 1)

    for (T_C ic = 0; ic < IC / PI; ic++) {
#pragma HLS PIPELINE
        for (T_C pi = 0; pi < PI; pi++) {
#pragma HLS UNROLL
            psum_buffer[ic * PI + pi] = 0;
        }
    }

    for (ap_uint<HW_W> rep = 0; rep < HEIGHT * WIDTH + 2; rep++) {
        T_K token = token_in.read();
        token_out.write(token);
        if (token.end == 1) break;
        for (ap_uint<4> k = 0; k < 10; k++) {
            T_OFFSET offset = offset_s.read();
            // cout << "offset:" << offset << endl;
            if (offset == end_3x3) break;
            for (T_C ic = 0; ic < IC / PI; ic++) {
#pragma HLS PIPELINE II = 1
                act_window = act_in.read();
                for (T_C pi = 0; pi < PI; pi++) {
#pragma HLS UNROLL
                    ap_int<AW> activation = (ap_int<AW>)(act_window.data[pi]);
                    ap_int<WW> weight = (ap_int<WW>)(w_buffer[offset][ic].range(
                        (pi + 1) * WW - 1, pi * WW));
                    psum_buffer[ic * PI + pi] += activation * weight;
                }
            }
        }
        for (T_C ic = 0; ic < IC / PI; ic++) {
#pragma HLS PIPELINE II = 1
            for (T_C pi = 0; pi < PI; pi++) {
#pragma HLS UNROLL
                psum_pack.data[pi] = psum_buffer[ic * PI + pi];
                psum_buffer[ic * PI + pi] = 0;
            }
            psum_out.write(psum_pack);
        }
    }
}

// // average pooling layer
// template <int PI, int PO, int IC, int N_CLASS, int HEIGHT, int WIDTH, int AW,
//           int WW, int PSUMW>
// void global_avgpool_linear(hls::stream<BundleT<PI, ap_int<AW>>> &act_in,
//                            ap_int<PO * PSUMW> *c_out,
//                            hls::stream<T_K> &token_in,
//                            const ap_int<PI * WW> weight[N_CLASS][IC / PI]) {
//     ap_int<PSUMW * PI> sum[IC / PI];
//     ap_int<PSUMW *PO> out_pack = 0;

//     for (int i = 0; i < IC / PI; i++) {
// #pragma HLS PIPELINE II = 1
//         sum[i] = 0;
//     }

//     for (int i = 0; i < HEIGHT * WIDTH + 2; i++) {
//         T_K token = token_in.read();
//         cout<<"token:"<<token.x<<" "<<token.y<<" "<<token.end<<endl;
//         if (token.end == 1) break;
//         for (int j = 0; j < IC / PI; j++) {
// #pragma HLS PIPELINE II = 1
//             BundleT<PI, ap_int<AW>> act = act_in.read();
//             ap_int<PSUMW *PI> s_pack = sum[j];
//             for (int pi = 0; pi < PI; pi++) {
//                 ap_int<PSUMW> s = (ap_int<PSUMW>)s_pack.range(
//                     PSUMW * (pi + 1) - 1, PSUMW * pi);
//                 ap_int<AW> a = act.data[pi];
//                 s += a;
//                 s_pack.range(PSUMW * (pi + 1) - 1, PSUMW * pi) = s;
//             }
//             sum[j] = s_pack;
//         }
//     }

//     // for (int i = 0; i < IC / PI; i++) {
//     //     for (int j = 0; j < PI; j++) {
//     //         cout << sum[i].range(PSUMW * (j + 1) - 1, PSUMW * j) << " ";
//     //     }
//     //     cout << endl;
//     // }

//     for (int oc = 0; oc < N_CLASS / PO; oc++) {
//         for (int ic = 0; ic < IC / PI; ic++) {
// #pragma HLS PIPELINE II = 1
//             for (int po = 0; po < PO; po++) {
//                 ap_int<PSUMW> logit = (ap_int<PSUMW>)out_pack.range(
//                     PSUMW * (po + 1) - 1, PSUMW * po);
//                 for (int pi = 0; pi < PI; pi++) {
//                     ap_int<PSUMW> s = (ap_int<PSUMW>)sum[ic].range(
//                         PSUMW * (pi + 1) - 1, PSUMW * pi);
//                     ap_int<AW> w = weight[oc * PO + po][ic].range(
//                         AW * (pi + 1) - 1, AW * pi);
//                     logit += s * w;
//                 }
//                 out_pack.range(PSUMW * (po + 1) - 1, PSUMW * po) = logit;
//             }
//         }
//         c_out[oc] = out_pack;
//         out_pack = 0;
//     }
// }

// PI == IC
template <int PI, int PO, int OC, int HEIGHT, int WIDTH, int AW, int WW,
          int PSUMW>
void conv_3x3_kernel_dsp_first_layer(
    hls::stream<BundleT<9, ap_uint<AW>>> &act_in,
    hls::stream<BundleT<PO, ap_int<PSUMW>>> &psum_out,
    hls::stream<T_K> &token_in, hls::stream<T_K> &token_out,
    const ap_int<WW> w_buffer[9][OC]) {
    const int C_W = boost::static_log2<OC>::value + 2;
    const int HW_W = boost::static_log2<HEIGHT * WIDTH>::value + 2;

// #pragma HLS bind_storage variable=w_buffer type=rom_2p impl=BRAM
#pragma HLS bind_storage variable = w_buffer type = rom_2p impl = LUTRAM

#pragma HLS ARRAY_PARTITION variable = w_buffer complete dim = 1
    DO_PRAGMA(HLS ARRAY_PARTITION variable = w_buffer cyclic factor =
                  (PO / 2) dim = 2)

    BundleT<PO * 9, ap_int<WW>> weight_pack;
#pragma HLS ARRAY_PARTITION variable = weight_pack.data complete dim = 0

    BundleT<9, ap_uint<AW>> act_in_pack;
#pragma HLS ARRAY_PARTITION variable = act_in_pack.data complete dim = 0

    ap_int<PSUMW> psum_buffer[OC];
#pragma HLS ARRAY_PARTITION variable = psum_buffer complete dim = 0

    typedef ap_uint<AW> T_ACT;
    typedef ap_int<WW> T_WEIGHT;
    typedef ap_int<PSUMW> T_PSUM;
    typedef ap_uint<C_W> T_C;

    BundleT<PO, ap_int<PSUMW>> psum_pack;
#pragma HLS ARRAY_PARTITION variable = psum_pack.data complete dim = 0

    for (T_C oc = 0; oc < OC / PO; oc++) {
        for (T_C po = 0; po < PO; po++) {
#pragma HLS UNROLL
            psum_buffer[oc * PO + po] = 0;
        }
    }

    // cout<<"WW:"<<WW<<" AW:"<<AW<<" PI:"<<PI<<endl;

    for (ap_uint<HW_W> r = 0; r < HEIGHT * WIDTH + 2; r++) {
        T_K token = token_in.read();
        token_out.write(token);
        if (token.end == 1) break;

        act_in_pack = act_in.read();

        for (T_C oc = 0; oc < OC / PO; oc++) {
#pragma HLS PIPELINE

            for (T_C po = 0; po < PO; po++) {
                for (ap_uint<4> k = 0; k < 9; k++) {
                    weight_pack.data[k + po * 9] = w_buffer[k][oc * PO + po];
                }
            }

            for (T_C po = 0; po < PO / 2; po++) {
                T_PSUM psum_0 = 0;
                T_PSUM psum_1 = 0;
                for (ap_uint<4> k = 0; k < 9; k++) {
                    T_ACT activation = act_in_pack.data[k];
                    cout<<" act:"<<activation;
                    ap_int<18> in_expend = (ap_int<18>)activation;

                    T_WEIGHT w_0 = weight_pack.data[po * 18 + k];
                    T_WEIGHT w_1 = weight_pack.data[po * 18 + 9 + k];

                    cout<<" w_0:"<<w_0<<" w_1:"<<w_1;

                    ap_int<27> w_1_shift = 0;
                    ap_int<27> w_0_expend = (ap_int<27>)w_0;
                    w_1_shift.range(26, 18) = (ap_int<9>)w_1;

                    ap_int<48> mul_temp =
                        DSP_AM(w_1_shift, w_0_expend, in_expend);
                    ap_int<AW + WW> low =
                        (ap_int<AW + WW>)(mul_temp.range(AW + WW - 1, 0));
                    ap_int<AW + WW> high =
                        (ap_int<AW + WW>)(mul_temp.range(AW + WW - 1 + 18,
                                                         18)) +
                        mul_temp.range(AW + WW - 1, AW + WW - 1);
                    psum_0 += low;
                    psum_1 += high;
                    cout<<" low:"<<low<<" high:"<<high<<endl;
                    cout<<" psum_0:"<<psum_0<<" psum_1:"<<psum_1<<endl;
                }
                psum_pack.data[po * 2] = psum_0;
                psum_pack.data[po * 2 + 1] = psum_1;
                cout<<" psum_0:"<<psum_0<<" psum_1:"<<psum_1<<endl;
            }
            psum_out.write(psum_pack);
        }
//         for (T_C oc = 0; oc < OC / PO; oc++) {
// #pragma HLS PIPELINE
//             for (T_C po = 0; po < PO; po++) {
//                 psum_pack.data[po] = psum_buffer[oc * PO + po];
//                 psum_buffer[oc * PO + po] = 0;
//             }
//             psum_out.write(psum_pack);
//         }
    }
}

// average pooling layer
template <int PI, int PO, int IC, int N_CLASS, int HEIGHT, int WIDTH, int AW,
          int WW, int EXP>
void global_avgpool_linear(hls::stream<BundleT<PI, ap_int<AW>>> &act_in,
                           hls::stream<T_K> &token_in, ap_int<32> *c_out,
                           const ap_int<PI * WW> weight[N_CLASS][IC / PI]) {
    static const int OUT_W = boost::static_log2<IC>::value + AW + WW +
                             boost::static_log2<HEIGHT * WIDTH>::value;
    static const int SUM_W = AW + boost::static_log2<HEIGHT * WIDTH>::value;

    static const int IC_W = boost::static_log2<IC>::value + 2;
    static const int OC_W = boost::static_log2<N_CLASS>::value + 2;
    static const int HW_W = boost::static_log2<HEIGHT * WIDTH>::value + 2;

    ap_int<SUM_W * PI> sum[IC / PI];

    ap_int<OUT_W> out_buffer[N_CLASS];
    DO_PRAGMA(HLS ARRAY_PARTITION variable = out_buffer cyclic factor = PO dim =
                  1)

#pragma HLS BIND_STORAGE variable = weight type = rom_2p impl = bram
    DO_PRAGMA(HLS ARRAY_PARTITION variable = weight cyclic factor = PO / 2 dim =
                                                                        1)

    for (ap_uint<IC_W> i = 0; i < IC / PI; i++) {
#pragma HLS PIPELINE II = 1
        sum[i] = 0;
    }

    for (ap_uint<OC_W> i = 0; i < N_CLASS; i++) {
#pragma HLS PIPELINE II = 1
        out_buffer[i] = 0;
    }

    for (ap_uint<HW_W> i = 0; i < HEIGHT * WIDTH + 1; i++) {
        T_K token = token_in.read();
        if (token.end == 1) break;
        for (ap_uint<IC_W> j = 0; j < IC / PI; j++) {
#pragma HLS PIPELINE II = 1
            BundleT<PI, ap_int<AW>> act = act_in.read();
            ap_int<SUM_W *PI> s_pack = sum[j];
            for (ap_uint<IC_W> pi = 0; pi < PI; pi++) {
                ap_int<SUM_W> s = (ap_int<SUM_W>)s_pack.range(
                    SUM_W * (pi + 1) - 1, SUM_W * pi);
                ap_int<AW> a = act.data[pi];
                s += a;
                s_pack.range(SUM_W * (pi + 1) - 1, SUM_W * pi) = s;
            }
            sum[j] = s_pack;
        }
    }

    for (ap_uint<OC_W> oc = 0; oc < N_CLASS / PO; oc++) {
        for (ap_uint<IC_W> ic = 0; ic < IC / PI; ic++) {
#pragma HLS PIPELINE II = 1
            for (ap_uint<OC_W> po = 0; po < PO; po++) {
                // ap_int<OUT_W> logit = out_buffer[oc * PO + po];
                ap_int<OUT_W> logit = 0;
                for (ap_uint<IC_W> pi = 0; pi < PI; pi++) {
                    ap_int<SUM_W> s = (ap_int<SUM_W>)sum[ic].range( SUM_W * (pi + 1) - 1, SUM_W * pi);
                    cout << "s:" << s << endl;
                    cout << "pi:" << pi << endl;
                    cout << "ic:" << ic << endl;
                    cout << "oc:" << oc << endl;
                    ap_int<WW> w = weight[oc * PO + po][ic].range( WW * (pi + 1) - 1, WW * pi);
                    cout << "w:" << w << endl;
                    out_buffer[oc * PO + po] += s * w;
                    cout << "out_buffer:" << out_buffer[oc * PO + po] << endl;
                }
            }
        }
    }

    for (int i = 0; i < N_CLASS; i++) {
        cout << "out_buffer:" << out_buffer[i] << endl;
    }

    for (int i = 0; i < N_CLASS; i++) {
#pragma HLS PIPELINE II = 1
        c_out[i] = out_buffer[i];
    }
}