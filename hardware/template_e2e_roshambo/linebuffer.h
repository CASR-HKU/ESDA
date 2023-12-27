

template <int PI, int IC, int HEIGHT, int WIDTH, int AW>
void conv_3x3_line_buffer_stride1_serial(
    hls::stream<BundleT<PI, ap_int<AW>>> &act_in,
    hls::stream<BundleT<PI, ap_int<AW>>> &act_out,
    hls::stream<T_OFFSET> &offset_s, hls::stream<T_K> &token_in,
    hls::stream<T_K> &token_out)
{
    static const int C_W = boost::static_log2<IC>::value + 2;
    static const int HW_W = boost::static_log2<HEIGHT * WIDTH>::value + 2;

    typedef ap_uint<C_W> T_C;
    typedef ap_int<HW_W> T_HW;

    const T_C ICPI = IC / PI;
    const T_K empty_token = {.end = 0, .x = 0, .y = 0};
    const int BUFFER_ROWS = 3;
    const int BUFFER_WIDTH = WIDTH * IC / PI;
    const int FIFO_DEPTH = WIDTH * (BUFFER_ROWS);

    bool token_read_enable = 1; 
    bool data_read_enable = 1;
    bool output_valid = 0;
    bool out_valid_one_line = 0;
    bool out_valid_multi_line = 0;

    ap_uint<HW_W>
        ptr_to_lastest = 0,
        ptr_to_oldest = 0; 
    T_K lastest_token = {.end = 0, .x = 0, .y = 0},
        oldest_token = {.end = 0, .x = 0, .y = 0};
    T_K oldest_token_reg = {.end = 0, .x = 0, .y = 0};

    T_HW y_delta = 0, jump_y = 0;

    T_K token_fifo[FIFO_DEPTH];
    ap_int<PI * AW> line_buff[BUFFER_ROWS][BUFFER_WIDTH];
    ap_int<PI *AW> output_read = 0;

    BundleT<PI, ap_int<AW>> output_pack;

    bool valid[BUFFER_ROWS][WIDTH];
#pragma HLS ARRAY_RESHAPE variable = valid complete dim = 2

    for (int i = 0; i < FIFO_DEPTH; i++)
    {
#pragma HLS PIPELINE
        token_fifo[i] = empty_token;
    }

    for (int r = 0; r < BUFFER_ROWS; r++)
    {
        for (int h = 0; h < BUFFER_WIDTH; h++)
        {
#pragma HLS PIPELINE
            line_buff[r][h] = 0;
        }
    }

    for (ap_uint<4> r = 0; r < BUFFER_ROWS; r++)
    {
#pragma HLS UNROLL
        for (T_HW h = 0; h < WIDTH; h++)
        {
#pragma HLS UNROLL
            valid[r][h] = 0;
        }
    }

    for (T_HW rep = 0; rep < HEIGHT * WIDTH * 2; rep++)
    {
        if (token_read_enable)
        { 
            lastest_token = token_in.read();
            token_fifo[ptr_to_lastest] = lastest_token;
            ptr_to_lastest = (ptr_to_lastest + 1) % FIFO_DEPTH;
        } 
        oldest_token_reg = token_fifo[ptr_to_oldest];
        jump_y = oldest_token_reg.y - oldest_token.y;

        if (jump_y > 0)
        {
            jump_y = jump_y > 3 ? (T_HW)3 : jump_y;
            for (T_HW l = 0; l < jump_y; l++)
            {
#pragma HLS PIPELINE II = 1

                for (T_HW k = 0; k < WIDTH; k++)
                {
#pragma HLS UNROLL
                    valid[(oldest_token.y + l + 2) % BUFFER_ROWS][k] = 0;
                }
            }
        }

        oldest_token = oldest_token_reg;
        y_delta = lastest_token.y - oldest_token.y;

        out_valid_one_line =
            (y_delta == 1) && (lastest_token.x - oldest_token.x >= 1);
        out_valid_multi_line = (y_delta >= 2);
        output_valid =
            out_valid_one_line || out_valid_multi_line || lastest_token.end;

        data_read_enable = (y_delta <= 1) && !lastest_token.end;
        token_read_enable =
            data_read_enable; // if data read in current iteration, then can
                              // start to read token in next iteration


        if (data_read_enable)
        {
            valid[lastest_token.y % BUFFER_ROWS][lastest_token.x] = 1;
        }


        if (output_valid)
        {
            token_out.write(oldest_token);
            ptr_to_oldest = (ptr_to_oldest + 1) % FIFO_DEPTH;
            if (oldest_token.end)
                break;
        }

        if (data_read_enable)
        {
            for (T_C ic = 0; ic < ICPI; ic++)
            {
#pragma HLS PIPELINE II = 1
                BundleT<PI, ap_int<AW>> data_in = act_in.read();
                ap_int<PI *AW> data_repacked = 0;
                for (T_C i = 0; i < PI; i++)
                {
                    data_repacked((i + 1) * AW - 1, i * AW) = data_in.data[i];
                }
                line_buff[lastest_token.y % BUFFER_ROWS]
                         [lastest_token.x * ICPI + ic] = data_repacked;
            }
        }

        if (output_valid)
        {
            for (ap_int<4> ki = -1; ki < 2; ki++)
            { // row
                for (ap_int<4> kj = -1; kj < 2; kj++)
                { // col
                    bool not_padding = (ki + oldest_token.y >= 0) &&
                                       (ki + oldest_token.y < HEIGHT) &&
                                       (kj + oldest_token.x >= 0) &&
                                       (kj + oldest_token.x < WIDTH);
                    bool valid_point = 0;
                    if (not_padding)
                    {
                        valid_point = valid[(oldest_token.y + ki) % BUFFER_ROWS]
                                           [oldest_token.x + kj];
                    }
                    if (valid_point)
                    {
                        T_OFFSET offset = (ki + 1) * 3 + kj + 1;
                        offset_s.write(offset);
                        for (T_C ic = 0; ic < ICPI; ic++)
                        {
#pragma HLS PIPELINE II = 1
                            output_read =
                                line_buff[(oldest_token.y + ki) % BUFFER_ROWS]
                                         [(oldest_token.x + kj) * ICPI + ic];
                            for (T_C pi = 0; pi < PI; pi++)
                            {
                                output_pack.data[pi] = output_read.range(
                                    (pi + 1) * AW - 1, pi * AW);
                            }
                            act_out.write(output_pack);
                        }
                    }
                }
            }
            offset_s.write(end_3x3);
        }
    }
}

template <int PI, int IC, int HEIGHT, int WIDTH, int AW>
void conv_3x3_line_buffer_stride2_fifo_serial(
    hls::stream<BundleT<PI, ap_int<AW>>> &act_in,
    hls::stream<BundleT<PI, ap_int<AW>>> &act_out, hls::stream<T_K> &token_in,
    hls::stream<T_K> &token_out, hls::stream<T_OFFSET> &offset_s)
{
    static const int C_W = boost::static_log2<IC>::value + 2;
    static const int HW_W = boost::static_log2<HEIGHT * WIDTH>::value + 2;

    typedef ap_uint<C_W> T_C;
    typedef ap_int<HW_W> T_HW;

    const T_C ICPI = IC / PI;
    const T_K empty_token = {.end = 0, .x = 0, .y = 0};
    const int BUFFER_ROWS = 3;
    const int BUFFER_WIDTH = WIDTH * IC / PI;
    const int FIFO_DEPTH = WIDTH * (BUFFER_ROWS);

    bool token_read_enable = 1; 
    bool data_read_enable = 1;
    bool output_valid = 0;
    bool out_valid_one_line = 0;
    bool out_valid_multi_line = 0;

    ap_uint<HW_W> even_ptr_to_lastest = 0, even_ptr_to_oldest = 0;
    ap_uint<HW_W> odd_ptr_to_lastest = 0, odd_ptr_to_oldest = 0;

    T_K even_lastest_token = {.end = 0, .x = 0, .y = 0},
        odd_lastest_token = {.end = 0, .x = 0, .y = 0};

    T_K lastest_token = {.end = 0, .x = 0, .y = 0};
    T_K oldest_token = {.end = 0, .x = 0, .y = 0};
    T_K oldest_token_reg = {.end = 0, .x = 0, .y = 0};
    T_K olddest_even_token_reg = {.end = 0, .x = 0, .y = 0};
    T_K olddest_odd_token_reg = {.end = 0, .x = 0, .y = 0};

    T_HW y_delta = 0, jump_y = 0;

    T_K even_token_fifo[FIFO_DEPTH];
    T_K odd_token_fifo[FIFO_DEPTH];
    bool pop_even = 0, pop_odd = 0;

    ap_uint<CFG_TW> new_x_s2, new_y_s2;
    ap_uint<CFG_TW> even_last_x_s2, even_last_y_s2, odd_last_x_s2,
        odd_last_y_s2;

    bool even_fifo_empty = 1, odd_fifo_empty = 1;

    T_K new_token;

    ap_int<PI * AW> line_buff[BUFFER_ROWS][BUFFER_WIDTH];
    ap_int<PI *AW> output_read = 0;
    BundleT<PI, ap_int<AW>> output_pack;

    bool valid[BUFFER_ROWS][WIDTH];
#pragma HLS ARRAY_RESHAPE variable = valid complete dim = 2

    bool win_valid[3][3];
#pragma HLS ARRAY_PARTITION variable = win_valid complete dim = 0

    BundleT<9, ap_int<PI * AW>> win;
#pragma HLS ARRAY_PARTITION variable = win.data complete dim = 0

    for (int i = 0; i < FIFO_DEPTH; i++)
    {
#pragma HLS PIPELINE
        even_token_fifo[i] = empty_token;
        odd_token_fifo[i] = empty_token;
    }

    for (ap_uint<4> r = 0; r < 3; r++)
    {
#pragma HLS UNROLL
        for (ap_uint<4> h = 0; h < 3; h++)
        {
#pragma HLS UNROLL
            win_valid[r][h] = 0;
        }
    }

    for (int r = 0; r < BUFFER_ROWS; r++)
    {
        for (int h = 0; h < BUFFER_WIDTH; h++)
        {
#pragma HLS PIPELINE
            line_buff[r][h] = 0;
        }
    }

    for (ap_uint<4> r = 0; r < BUFFER_ROWS; r++)
    {
#pragma HLS UNROLL
        for (T_HW h = 0; h < WIDTH; h++)
        {
#pragma HLS UNROLL
            valid[r][h] = 0;
        }
    }

    for (T_HW rep = 0; rep < HEIGHT * WIDTH * 2; rep++)
    {
        if (token_read_enable)
        {
            new_token = token_in.read();
            lastest_token = new_token;
            new_x_s2 = new_token.x >> 1;
            new_y_s2 = new_token.y >> 1;
            even_last_x_s2 = even_lastest_token.x >> 1;
            even_last_y_s2 = even_lastest_token.y >> 1;
            odd_last_x_s2 = odd_lastest_token.x >> 1;
            odd_last_y_s2 = odd_lastest_token.y >> 1;
            if (new_token.y[0] == 0)
            { // even case
                if (new_x_s2 != even_last_x_s2 || new_y_s2 != even_last_y_s2 ||
                    even_fifo_empty || new_token.end)
                {
                    even_token_fifo[even_ptr_to_lastest] = new_token;

                    even_lastest_token = new_token;
                    even_ptr_to_lastest =
                        (even_ptr_to_lastest + 1) % FIFO_DEPTH;
                }
            }
            else
            { // odd
                if (new_x_s2 != odd_last_x_s2 || new_y_s2 != odd_last_y_s2 ||
                    odd_fifo_empty || new_token.end)
                {
                    odd_token_fifo[odd_ptr_to_lastest] = new_token;
                    odd_lastest_token = new_token;
                    odd_ptr_to_lastest = (odd_ptr_to_lastest + 1) % FIFO_DEPTH;
                }
            }
        }

        olddest_even_token_reg = even_token_fifo[even_ptr_to_oldest];
        olddest_odd_token_reg = odd_token_fifo[odd_ptr_to_oldest];

        olddest_even_token_reg.x[0] = 0;
        olddest_even_token_reg.y[0] = 0;
        olddest_odd_token_reg.x[0] = 0;
        olddest_odd_token_reg.y[0] = 0;

        ap_uint<20> key_even =
            olddest_even_token_reg.x + olddest_even_token_reg.y * WIDTH;
        ap_uint<20> key_odd =
            olddest_odd_token_reg.x + olddest_odd_token_reg.y * WIDTH;

        even_fifo_empty = (even_ptr_to_lastest == even_ptr_to_oldest);
        odd_fifo_empty = (odd_ptr_to_lastest == odd_ptr_to_oldest);


        if (((key_even < key_odd) && !even_fifo_empty) || odd_fifo_empty)
        {
            oldest_token_reg = olddest_even_token_reg;
            pop_even = 1;
            pop_odd = 0;
        }
        else if (((key_even > key_odd) && !odd_fifo_empty) ||
                 even_fifo_empty)
        {
            oldest_token_reg = olddest_odd_token_reg;
            pop_odd = 1;
            pop_even = 0;
        }
        else
        {
            oldest_token_reg = olddest_odd_token_reg;
            pop_odd = 1;
            pop_even = 1;
        }

        jump_y = oldest_token_reg.y - oldest_token.y;

        if (jump_y > 0)
        {
            jump_y = jump_y > 3 ? (T_HW)3 : jump_y;
            for (T_HW l = 0; l < jump_y; l++)
            {
#pragma HLS PIPELINE II = 1

                for (T_HW k = 0; k < WIDTH; k++)
                {
#pragma HLS UNROLL
                    valid[(oldest_token.y + l + 2) % BUFFER_ROWS][k] = 0;
                }
            }
        }

        oldest_token = oldest_token_reg;
        y_delta = lastest_token.y - oldest_token.y;

        out_valid_one_line =
            (y_delta == 1) && (lastest_token.x - oldest_token.x >= 1);
        out_valid_multi_line = (y_delta >= 2);
        output_valid =
            out_valid_one_line || out_valid_multi_line || lastest_token.end;

        data_read_enable = (y_delta <= 1) && !lastest_token.end;
        token_read_enable = data_read_enable; 


        if (data_read_enable)
        {
            valid[lastest_token.y % BUFFER_ROWS][lastest_token.x] = 1;
        }



        if (output_valid)
        {
            T_K stride2_token;
            stride2_token.x = oldest_token.x >> 1;
            stride2_token.y = oldest_token.y >> 1;
            stride2_token.end = oldest_token.end;

            token_out.write(stride2_token);

            if (pop_even)
                even_ptr_to_oldest = (even_ptr_to_oldest + 1) % FIFO_DEPTH;
            if (pop_odd)
                odd_ptr_to_oldest = (odd_ptr_to_oldest + 1) % FIFO_DEPTH;
            if (oldest_token.end)
                break;
        }

        if (data_read_enable)
        {
            for (T_C ic = 0; ic < ICPI; ic++)
            {
#pragma HLS PIPELINE
                BundleT<PI, ap_int<AW>> data_in = act_in.read();
                ap_int<PI *AW> data_repacked = 0;
                for (T_C i = 0; i < PI; i++)
                {
                    data_repacked((i + 1) * AW - 1, i * AW) = data_in.data[i];
                }
                line_buff[lastest_token.y % BUFFER_ROWS]
                         [lastest_token.x * ICPI + ic] = data_repacked;
            }
        }

        if (output_valid)
        {
            for (ap_int<4> ki = -1; ki < 2; ki++)
            { // row
                for (ap_int<4> kj = -1; kj < 2; kj++)
                { // col
                    bool not_padding = (ki + oldest_token.y >= 0) &&
                                       (ki + oldest_token.y < HEIGHT) &&
                                       (kj + oldest_token.x >= 0) &&
                                       (kj + oldest_token.x < WIDTH);
                    bool valid_point = 0;
                    if (not_padding)
                    {
                        valid_point = valid[(oldest_token.y + ki) % BUFFER_ROWS]
                                           [oldest_token.x + kj];
                    }
                    if (valid_point)
                    {
                        T_OFFSET offset = (ki + 1) * 3 + kj + 1;
                        offset_s.write(offset);
                        for (T_C ic = 0; ic < ICPI; ic++)
                        {
#pragma HLS PIPELINE II = 1
                            output_read =
                                line_buff[(oldest_token.y + ki) % BUFFER_ROWS]
                                         [(oldest_token.x + kj) * ICPI + ic];
                            for (T_C pi = 0; pi < PI; pi++)
                            {
                                output_pack.data[pi] = output_read.range(
                                    (pi + 1) * AW - 1, pi * AW);
                            }
                            act_out.write(output_pack);
                        }
                    }
                }
            }
            offset_s.write(end_3x3);
        }
    }
}

template <int PI, int IC, int HEIGHT, int WIDTH, int AW>
void conv_3x3_line_buffer_first_layer(
    hls::stream<ap_uint<AW> > &act_in,
    hls::stream<BundleT<9, ap_uint<AW>>> &act_out,
    hls::stream<T_K> &token_in, hls::stream<T_K> &token_out,
    hls::stream<T_K> &token_stride2)
{
    static const int C_W = boost::static_log2<IC>::value + 2;
    static const int HW_W = boost::static_log2<HEIGHT * WIDTH>::value + 2;

    typedef ap_uint<C_W> T_C;
    typedef ap_int<HW_W> T_HW;

    const T_C ICPI = IC;
    const T_K empty_token = {.end = 0, .x = 0, .y = 0};
    const int BUFFER_ROWS = 3;
    const int BUFFER_WIDTH = WIDTH * IC;
    const int FIFO_DEPTH = WIDTH * (BUFFER_ROWS);

    bool token_read_enable = 1; 
    bool data_read_enable = 1;
    bool output_valid = 0;
    bool out_valid_one_line = 0;
    bool out_valid_multi_line = 0;

    ap_uint<HW_W> even_ptr_to_lastest = 0, even_ptr_to_oldest = 0;
    ap_uint<HW_W> odd_ptr_to_lastest = 0, odd_ptr_to_oldest = 0;

    T_K even_lastest_token = {.end = 0, .x = 0, .y = 0},
        odd_lastest_token = {.end = 0, .x = 0, .y = 0};

    T_K lastest_token = {.end = 0, .x = 0, .y = 0};
    T_K oldest_token = {.end = 0, .x = 0, .y = 0};
    T_K oldest_token_reg = {.end = 0, .x = 0, .y = 0};
    T_K olddest_even_token_reg = {.end = 0, .x = 0, .y = 0};
    T_K olddest_odd_token_reg = {.end = 0, .x = 0, .y = 0};

    T_HW y_delta = 0, jump_y = 0;

    T_K even_token_fifo[FIFO_DEPTH];
    T_K odd_token_fifo[FIFO_DEPTH];
    bool pop_even = 0, pop_odd = 0;

    ap_uint<CFG_TW> new_x_s2, new_y_s2;
    ap_uint<CFG_TW> even_last_x_s2, even_last_y_s2, odd_last_x_s2,
        odd_last_y_s2;

    bool even_fifo_empty = 1, odd_fifo_empty = 1;

    T_K new_token;

    ap_uint<AW> line_buff[BUFFER_ROWS][BUFFER_WIDTH];
#pragma HLS ARRAY_PARTITION variable = line_buff complete dim = 1

    ap_uint<WIDTH> valid[BUFFER_ROWS];
#pragma HLS ARRAY_PARTITION variable = valid complete dim = 1

    bool win_valid[3][3];
#pragma HLS ARRAY_PARTITION variable = win_valid complete dim = 0

    BundleT<9, ap_uint<AW>> win;
#pragma HLS ARRAY_PARTITION variable = win.data complete dim = 0

    for (int i = 0; i < FIFO_DEPTH; i++)
    {
#pragma HLS PIPELINE
        even_token_fifo[i] = empty_token;
        odd_token_fifo[i] = empty_token;
    }

    for (ap_uint<4> r = 0; r < 3; r++)
    {
#pragma HLS UNROLL
        for (ap_uint<4> h = 0; h < 3; h++)
        {
#pragma HLS UNROLL
            win_valid[r][h] = 0;
        }
    }

    for (T_HW h = 0; h < BUFFER_WIDTH; h++)
    {
#pragma HLS PIPELINE
        for (ap_uint<4> r = 0; r < BUFFER_ROWS; r++)
        {
#pragma HLS UNROLL
            line_buff[r][h] = 0;
        }
    }

    for (T_HW r = 0; r < BUFFER_ROWS; r++)
    {
#pragma HLS UNROLL
        valid[r] = 0;
    }

    bool first_flag = 1;
    bool token_stride2_read_enable = 1;

    for (T_HW rep = 0; rep < HEIGHT * WIDTH * 2; rep++)
    {
#pragma HLS PIPELINE
        if (token_read_enable)
        {
            new_token = token_in.read();
            lastest_token = new_token;
        }

        if (token_stride2_read_enable)
        {
            oldest_token_reg = token_stride2.read();
            jump_y = oldest_token_reg.y - oldest_token.y;
            if (jump_y >= 3)
            {
                valid[0] = 0;
                valid[1] = 0;
                valid[2] = 0;
            }
            else if (jump_y == 2)
            {
                valid[(oldest_token.y + 2) % BUFFER_ROWS] = 0;
                valid[(oldest_token.y) % BUFFER_ROWS] = 0;
            }
            else if (jump_y == 1)
            {
                valid[(oldest_token.y + 2) % BUFFER_ROWS] = 0;
            }
            oldest_token = oldest_token_reg;
        }

        y_delta = lastest_token.y - oldest_token.y;

        out_valid_one_line =
            (y_delta == 1) && (lastest_token.x - oldest_token.x >= 1);
        out_valid_multi_line = (y_delta >= 2);
        output_valid =
            out_valid_one_line || out_valid_multi_line || lastest_token.end;

        data_read_enable = (y_delta <= 1) && !lastest_token.end;
        token_read_enable =  data_read_enable; 


        if (data_read_enable)
        {
            valid[lastest_token.y % BUFFER_ROWS][lastest_token.x] = 1;
        }

        if (output_valid)
        {
            T_K stride2_token;
            stride2_token.x = oldest_token.x >> 1;
            stride2_token.y = oldest_token.y >> 1;
            stride2_token.end = oldest_token.end;
            token_out.write(stride2_token);
            if (oldest_token.end)
                break;
            token_stride2_read_enable = 1;
        }
        else
        {
            token_stride2_read_enable = 0;
        }

        if (data_read_enable)
        {
            ap_uint<AW> data_in = act_in.read();

            line_buff[lastest_token.y % BUFFER_ROWS][lastest_token.x] =
                data_in;
        }

        if (output_valid)
        {
            for (ap_int<4> ki = -1; ki < 2; ki++)
            { // row
#pragma HLS UNROLL
                for (ap_int<4> kj = -1; kj < 2; kj++)
                { // col
#pragma HLS UNROLL
                    bool not_padding = (ki + oldest_token.y >= 0) &&
                                       (ki + oldest_token.y < HEIGHT) &&
                                       (kj + oldest_token.x >= 0) &&
                                       (kj + oldest_token.x < WIDTH);
        
                    bool valid_point = 0;
                    if (not_padding)
                    {
                        valid_point = valid[(oldest_token.y + ki) % BUFFER_ROWS]
                                           [oldest_token.x + kj];
                    }

                    if (valid_point)
                    {
                        win.data[(ki + 1) * 3 + kj + 1] =
                            line_buff[(oldest_token.y + ki) % BUFFER_ROWS]
                                     [(oldest_token.x + kj)];
                    }
                    else
                    {
                        win.data[(ki + 1) * 3 + kj + 1] = 0;
                    }
                }
            }
            act_out.write(win);
        }
    }
}