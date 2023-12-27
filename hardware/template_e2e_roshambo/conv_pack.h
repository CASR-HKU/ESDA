template <int PI, int PO, int IC, int OC, int HEIGHT, int WIDTH, int AW, int WW,
          int PSUMW, int SCALEW, int BIASW, int EXP, bool RELU>
void conv_1x1_module(hls::stream<BundleT<PI, ap_int<AW>>> &act_in,
                     hls::stream<BundleT<PO, ap_int<AW>>> &act_out,
                     hls::stream<T_K> &token_in, hls::stream<T_K> &token_out,
                     const ap_int<PI * WW> w_buffer[OC][IC / PI],
                     const ap_int<SCALEW + BIASW> scale_buffer[OC]) {
#pragma HLS DATAFLOW
    hls::stream<BundleT<PO, ap_int<PSUMW>>> pusm_1;
#pragma HLS STREAM variable = pusm_1 depth = 2
    hls::stream<T_K> token_1;
#pragma HLS STREAM variable = token_1 depth = 2

    conv_1x1_kernel_dsp<PI, PO, IC, OC, HEIGHT, WIDTH, AW, WW, PSUMW>(
        act_in, pusm_1, token_in, token_1, w_buffer);

    // quantize
    quantize<PO, OC, HEIGHT, WIDTH, PSUMW, AW, SCALEW, BIASW, EXP, RELU>(
        pusm_1, act_out, token_1, token_out, scale_buffer);
}

template <int PI, int PO, int IC, int OC, int HEIGHT, int WIDTH, int AW, int WW,
          int PSUMW, int SCALEW, int BIASW, int SCALEIDW, int EXP>
void conv_1x1_module_residual(hls::stream<BundleT<PI, ap_int<AW>>> &act_in,
                              hls::stream<BundleT<PO, ap_int<AW>>> &act_id,
                              hls::stream<BundleT<PO, ap_int<AW>>> &act_out,
                              hls::stream<T_K> &token_in,
                              hls::stream<T_K> &token_out,
                              const ap_int<PI * WW> w_buffer[OC][IC / PI],
                              const ap_int<SCALEW + BIASW> scale_buffer[OC],
                              const ap_int<SCALEIDW> id_scale) {
#pragma HLS DATAFLOW
    hls::stream<BundleT<PO, ap_int<PSUMW>>> pusm_1;
#pragma HLS STREAM variable = pusm_1 depth = 2
    hls::stream<T_K> token_1;
#pragma HLS STREAM variable = token_1 depth = 2

    conv_1x1_kernel_dsp<PI, PO, IC, OC, HEIGHT, WIDTH, AW, WW, PSUMW>(
        act_in, pusm_1, token_in, token_1, w_buffer);

    // quantize
    quantize_id_add<PO, OC, HEIGHT, WIDTH, PSUMW, AW, SCALEW, BIASW, SCALEIDW,
                    EXP>(pusm_1, act_id, act_out, token_1, token_out,
                         scale_buffer, id_scale);
}

template <int PI, int IC, int HEIGHT, int WIDTH, int AW, int WW, int PSUMW,
          int SCALEW, int BIASW, int EXP>
void conv_3x3_dw_module_stride1_serial(
    hls::stream<BundleT<PI, ap_int<AW>>> &act_in,
    hls::stream<BundleT<PI, ap_int<AW>>> &act_out, hls::stream<T_K> &token_in,
    hls::stream<T_K> &token_out, const ap_int<PI * WW> w_buffer[9][IC / PI],
    const ap_int<SCALEW + BIASW> scale_buffer[IC]) {
#pragma HLS DATAFLOW
    //     hls::stream<BundleT<9, ap_int<PI * AW> > > win_0;
    // #pragma HLS STREAM variable=win_0 depth=2

    hls::stream<BundleT<PI, ap_int<AW>>> win_0;
#pragma HLS STREAM variable = win_0 depth = 8

    hls::stream<T_K> token_0;
#pragma HLS STREAM variable = token_0 depth = 2

    hls::stream<BundleT<PI, ap_int<PSUMW>>> pusm_1;
#pragma HLS STREAM variable = pusm_1 depth = 2
    hls::stream<T_K> token_1;
#pragma HLS STREAM variable = token_1 depth = 2

    hls::stream<T_OFFSET> offset_stream;
#pragma HLS STREAM variable = offset_stream depth = 9

    // linebuffer
    conv_3x3_line_buffer_stride1_serial<PI, IC, HEIGHT, WIDTH, AW>(
        act_in, win_0, offset_stream, token_in, token_0);

    // conv
    conv_3x3_dw_kernel_serial<PI, IC, HEIGHT, WIDTH, AW, WW, PSUMW>(
        win_0, pusm_1, token_0, token_1, offset_stream, w_buffer);

    // quantize
    quantize<PI, IC, HEIGHT, WIDTH, PSUMW, AW, SCALEW, BIASW, EXP, 1>(
        pusm_1, act_out, token_1, token_out, scale_buffer);
}

template <int PI, int IC, int HEIGHT, int WIDTH, int AW, int WW, int PSUMW,
          int SCALEW, int BIASW, int EXP>
void conv_3x3_dw_module_stride2_serial(
    hls::stream<BundleT<PI, ap_int<AW>>> &act_in,
    hls::stream<BundleT<PI, ap_int<AW>>> &act_out, hls::stream<T_K> &token_in,
    hls::stream<T_K> &token_out, const ap_int<PI * WW> w_buffer[9][IC / PI],
    const ap_int<SCALEW + BIASW> scale_buffer[IC]) {
#pragma HLS DATAFLOW
    hls::stream<BundleT<PI, ap_int<AW>>> win_0;
#pragma HLS STREAM variable = win_0 depth = 8
    hls::stream<T_K> token_0;
#pragma HLS STREAM variable = token_0 depth = 2

    hls::stream<BundleT<PI, ap_int<PSUMW>>> pusm_1;
#pragma HLS STREAM variable = pusm_1 depth = 2
    hls::stream<T_K> token_1;
#pragma HLS STREAM variable = token_1 depth = 2

    hls::stream<T_OFFSET> offset_0;
#pragma HLS STREAM variable = offset_0 depth = 9

    // linebuffer
    conv_3x3_line_buffer_stride2_fifo_serial<PI, IC, HEIGHT, WIDTH, AW>(
        act_in, win_0, token_in, token_0, offset_0);
    // conv_3x3_line_buffer_stride2_bitmap<PI, IC, HEIGHT, WIDTH, AW>(act_in,
    // win_0, token_in, token_0);

    // conv
    conv_3x3_dw_kernel_serial<PI, IC, HEIGHT, WIDTH, AW, WW, PSUMW>(
        win_0, pusm_1, token_0, token_1, offset_0, w_buffer);

    // quantize
    quantize<PI, IC, HEIGHT, WIDTH, PSUMW, AW, SCALEW, BIASW, EXP, 1>(
        pusm_1, act_out, token_1, token_out, scale_buffer);
}

template <int PF_0, int PF_1, int PF_2, int IC, int C, int OC, int HEIGHT,
          int WIDTH, int PSUMW0, int PSUMW1, int PSUMW2, int SCALEW0,
          int SCALEW1, int SCALEW2, int BIASW0, int BIASW1, int BIASW2, int AW,
          int WW, int EXP>
void conv_1x1_3x3_dw_1x1_stride1(
    hls::stream<BundleT<PF_0, ap_int<AW>>> &act_in,
    hls::stream<BundleT<PF_2, ap_int<AW>>> &act_out, hls::stream<T_K> &token_in,
    hls::stream<T_K> &token_out,
    const ap_int<PF_0 * WW> w_buffer_0[C][IC / PF_0],
    const ap_int<PF_1 * WW> w_buffer_1[9][C / PF_1],
    const ap_int<PF_1 * WW> w_buffer_2[OC][C / PF_1],
    const ap_int<SCALEW0 + BIASW0> scale_buffer_0[C],
    const ap_int<SCALEW1 + BIASW1> scale_buffer_1[C],
    const ap_int<SCALEW1 + BIASW1> scale_buffer_2[OC]) {
#pragma HLS Dataflow

    hls::stream<BundleT<PF_1, ap_int<AW>>> act_1;
#pragma HLS STREAM variable = act_1 depth = 2
    hls::stream<T_K> token_1;
#pragma HLS STREAM variable = token_1 depth = 16
    hls::stream<BundleT<PF_1, ap_int<AW>>> act_2;
#pragma HLS STREAM variable = act_2 depth = 2
    hls::stream<T_K> token_2;
#pragma HLS STREAM variable = token_2 depth = 16

    // conv 1x1
    // cout << "conv 1x1" << endl;
    conv_1x1_module<PF_0, PF_1, IC, C, HEIGHT, WIDTH, AW, WW, PSUMW0, SCALEW0,
                    BIASW0, EXP, 1>(act_in, act_1, token_in, token_1,
                                    w_buffer_0, scale_buffer_0);

    // conv 3x3 dw
    // cout << "conv 3x3 dw" << endl;
    conv_3x3_dw_module_stride1_serial<PF_1, C, HEIGHT, WIDTH, AW, WW, PSUMW1,
                                      SCALEW1, BIASW1, EXP>(
        act_1, act_2, token_1, token_2, w_buffer_1, scale_buffer_1);

    // conv 1x1
    // cout << "conv 1x1_2" << endl;
    conv_1x1_module<PF_1, PF_2, C, OC, HEIGHT, WIDTH, AW, WW, PSUMW2, SCALEW2,
                    BIASW2, EXP, 0>(act_2, act_out, token_2, token_out,
                                    w_buffer_2, scale_buffer_2);
}

template <int PF_0, int PF_1, int PF_2, int IC, int C, int OC, int HEIGHT,
          int WIDTH, int PSUMW0, int PSUMW1, int PSUMW2, int SCALEW0,
          int SCALEW1, int SCALEW2, int BIASW0, int BIASW1, int BIASW2, int AW,
          int WW, int EXP>
void conv_1x1_3x3_dw_1x1_stride2(
    hls::stream<BundleT<PF_0, ap_int<AW>>> &act_in,
    hls::stream<BundleT<PF_2, ap_int<AW>>> &act_out, hls::stream<T_K> &token_in,
    hls::stream<T_K> &token_out,
    const ap_int<PF_0 * WW> w_buffer_0[C][IC / PF_0],
    const ap_int<PF_1 * WW> w_buffer_1[9][C / PF_1],
    const ap_int<PF_1 * WW> w_buffer_2[OC][C / PF_1],
    const ap_int<SCALEW0 + BIASW0> scale_buffer_0[C],
    const ap_int<SCALEW1 + BIASW1> scale_buffer_1[C],
    const ap_int<SCALEW2 + BIASW2> scale_buffer_2[OC]) {
#pragma HLS Dataflow

    hls::stream<BundleT<PF_1, ap_int<AW>>> act_1;
#pragma HLS STREAM variable = act_1 depth = 2
    hls::stream<T_K> token_1;
#pragma HLS STREAM variable = token_1 depth = 2
    hls::stream<BundleT<PF_1, ap_int<AW>>> act_2;
#pragma HLS STREAM variable = act_2 depth = 2
    hls::stream<T_K> token_2;
#pragma HLS STREAM variable = token_2 depth = 2

    // conv 1x1
    // cout << "conv 1x1" << endl;
    conv_1x1_module<PF_0, PF_1, IC, C, HEIGHT, WIDTH, AW, WW, PSUMW0, SCALEW0,
                    BIASW0, EXP, 1>(act_in, act_1, token_in, token_1,
                                    w_buffer_0, scale_buffer_0);

    // conv 3x3 dw
    // cout << "conv 3x3 dw" << endl;
    conv_3x3_dw_module_stride2_serial<PF_1, C, HEIGHT, WIDTH, AW, WW, PSUMW1,
                                      SCALEW1, BIASW1, EXP>(
        act_1, act_2, token_1, token_2, w_buffer_1, scale_buffer_1);

    // conv 1x1
    // cout << "conv 1x1_2" << endl;
    conv_1x1_module<PF_1, PF_2, C, OC, HEIGHT, WIDTH, AW, WW, PSUMW2, SCALEW2,
                    BIASW2, EXP, 0>(act_2, act_out, token_2, token_out,
                                    w_buffer_2, scale_buffer_2);
}

template <int PF_0, int PF_1, int PF_2, int IC, int C, int HEIGHT, int WIDTH,
          int PSUMW0, int PSUMW1, int PSUMW2, int SCALEW0, int SCALEW1,
          int SCALEW2, int BIASW0, int BIASW1, int BIASW2, int SCALEIDW, int AW,
          int WW, int EXP>
void conv_1x1_3x3_dw_1x1_stride1_residual(
    hls::stream<BundleT<PF_0, ap_int<AW>>> &act_in,
    hls::stream<BundleT<PF_2, ap_int<AW>>> &act_out, hls::stream<T_K> &token_in,
    hls::stream<T_K> &token_out,
    const ap_int<PF_0 * WW> w_buffer_0[C][IC / PF_0],
    const ap_int<PF_1 * WW> w_buffer_1[9][C / PF_1],
    const ap_int<PF_1 * WW> w_buffer_2[IC][C / PF_1],
    const ap_int<SCALEW0 + BIASW0> scale_buffer_0[C],
    const ap_int<SCALEW1 + BIASW1> scale_buffer_1[C],
    const ap_int<SCALEW2 + BIASW2> scale_buffer_2[IC],
    const ap_int<SCALEIDW> id_scale) {
#pragma HLS DATAFLOW

    hls::stream<BundleT<PF_0, ap_int<AW>>> act_0;
#pragma HLS STREAM variable = act_0 depth = 2

    hls::stream<BundleT<PF_1, ap_int<AW>>> act_1;
#pragma HLS STREAM variable = act_1 depth = 2
    hls::stream<T_K> token_0;
#pragma HLS STREAM variable = token_0 depth = 2
    hls::stream<T_K> token_1;
#pragma HLS STREAM variable = token_1 depth = 2
    hls::stream<BundleT<PF_1, ap_int<AW>>> act_2;
#pragma HLS STREAM variable = act_2 depth = 2
    hls::stream<T_K> token_2;
#pragma HLS STREAM variable = token_2 depth = 2

    const int OC = IC;
    hls::stream<BundleT<PF_2, ap_int<AW>>> act_id;
    DO_PRAGMA(HLS STREAM variable = act_id depth = (WIDTH + 2) * OC / PF_2)
    hls::stream<T_K> id_token;
    DO_PRAGMA(HLS STREAM variable = id_token depth = (WIDTH + 2))

    duplicate_stream<PF_0, PF_2, IC, AW, HEIGHT, WIDTH>(act_in, act_0, act_id,
                                                        token_in, token_0);

    // cout << "conv 1x1" << endl;
    conv_1x1_module<PF_0, PF_1, IC, C, HEIGHT, WIDTH, AW, WW, PSUMW0, SCALEW0,
                    BIASW0, EXP, 1>(act_0, act_1, token_0, token_1, w_buffer_0,
                                    scale_buffer_0);

    // conv 3x3 dw
    // cout << "conv 3x3 dw" << endl;
    conv_3x3_dw_module_stride1_serial<PF_1, C, HEIGHT, WIDTH, AW, WW, PSUMW1,
                                      SCALEW1, BIASW1, EXP>(
        act_1, act_2, token_1, token_2, w_buffer_1, scale_buffer_1);

    // conv 1x1
    // cout << "conv 1x1_2" << endl;
    conv_1x1_module_residual<PF_1, PF_2, C, OC, HEIGHT, WIDTH, AW, WW, PSUMW2,
                             SCALEW2, BIASW2, SCALEIDW, EXP>(
        act_2, act_id, act_out, token_2, token_out, w_buffer_2, scale_buffer_2,
        id_scale);
}

// PI == IC
template <int PI, int PO, int OC, int HEIGHT, int WIDTH, int PSUMW, int SCALEW,
          int BIASW, int AW, int WW, int EXP>
void conv_3x3_first_layer(hls::stream<ap_uint<AW> > &act_in,
                          hls::stream<BundleT<PO, ap_int<AW>>> &act_out,
                          hls::stream<T_K> &token_in,
                          hls::stream<T_K> &token_out,
                          hls::stream<T_K> &token_stride2,
                          const ap_int<WW> w_buffer[9][OC],
                          const ap_int<SCALEW + BIASW> scale_buffer[OC]) {
#pragma HLS DATAFLOW

    hls::stream<BundleT<9, ap_uint<AW>>> win_0;
#pragma HLS STREAM variable = win_0 depth = 2
    hls::stream<T_K> token_0;
#pragma HLS STREAM variable = token_0 depth = 9

    hls::stream<BundleT<PO, ap_int<PSUMW>>> pusm_1;
#pragma HLS STREAM variable = pusm_1 depth = 2
    hls::stream<T_K> token_1;
#pragma HLS STREAM variable = token_1 depth = 2

    // linebuffer
    conv_3x3_line_buffer_first_layer<1, 1, HEIGHT, WIDTH, AW>(
        act_in, win_0, token_in, token_0, token_stride2);

    conv_3x3_kernel_dsp_first_layer<1, PO, OC, HEIGHT, WIDTH, AW, WW, PSUMW>(
        win_0, pusm_1, token_0, token_1, w_buffer);

    // quantize
    quantize<PO, OC, HEIGHT, WIDTH, PSUMW, AW, SCALEW, BIASW, EXP, 1>(
        pusm_1, act_out, token_1, token_out, scale_buffer);
}

template <int PI, int PO, int IC, int OC, int HEIGHT, int WIDTH, int PSUMW,
          int SCALEW, int BIASW, int AW, int WW, int EXP>
void conv8(hls::stream<BundleT<PI, ap_int<AW>>> &act_in,
           hls::stream<BundleT<PO, ap_int<AW>>> &act_out,
           hls::stream<T_K> &token_in, hls::stream<T_K> &token_out,
           const ap_int<PI * WW> w_buffer[OC][IC / PI],
           const ap_int<SCALEW + BIASW> scale_buffer[OC]) {
    conv_1x1_module<PI, PO, IC, OC, HEIGHT, WIDTH, AW, WW, PSUMW, SCALEW, BIASW,
                    EXP, 1>(act_in, act_out, token_in, token_out, w_buffer,
                            scale_buffer);
}
