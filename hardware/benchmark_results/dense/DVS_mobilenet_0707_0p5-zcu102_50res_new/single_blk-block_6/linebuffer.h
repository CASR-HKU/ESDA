

template <int PI, int IC, int HEIGHT, int WIDTH, int AW>
void conv_3x3_line_buffer_stride1_serial(
    hls::stream<BundleT<PI, ap_int<AW>>> &act_in,
    hls::stream<BundleT<PI, ap_int<AW>>> &act_out
)
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

 
    ap_int<PI * AW> line_buff[BUFFER_ROWS][BUFFER_WIDTH];
    ap_int<PI *AW> output_read = 0;

    BundleT<PI, ap_int<AW>> output_pack;

    bool valid[BUFFER_ROWS][WIDTH];
#pragma HLS ARRAY_RESHAPE variable = valid complete dim = 2

    for (int r = 0; r < BUFFER_ROWS; r++)
    {
        for (int h = 0; h < BUFFER_WIDTH; h++)
        {
#pragma HLS PIPELINE
            line_buff[r][h] = 0;
        }
    }

    int read_count = 0;
    int in_x = 0, in_y = 0;
    int out_x = 0, out_y = 0;
    for(T_HW h = -1; h < HEIGHT; h++){
        for(T_HW r = -1; r < WIDTH; r++){
            bool out_enable = (h >= 0) && (r >= 0);
            bool read_enable = read_count < HEIGHT * WIDTH;
            if (read_enable){
                for(T_C ic = 0; ic < ICPI; ic++){
#pragma HLS PIPELINE II = 1
                    BundleT<PI, ap_int<AW>> data_in = act_in.read();
                    ap_int<PI *AW> data_repacked = 0;
                    for (T_C i = 0; i < PI; i++)
                    {
                        data_repacked((i + 1) * AW - 1, i * AW) = data_in.data[i];
                    }
                    line_buff[in_y % BUFFER_ROWS][in_x * ICPI + ic] = data_repacked;
                }
                in_x++;
                if (in_x == WIDTH){
                    in_x = 0;
                    in_y++;
                }
                read_count++;
            }
            if (out_enable){
                for (ap_int<4> ki = -1; ki < 2; ki++)
                { // row
                    for (ap_int<4> kj = -1; kj < 2; kj++)
                    { // col
                        bool padding = h + ki < 0 || h + ki >= HEIGHT || r + kj < 0 || r + kj >= WIDTH;
                        for (T_C ic = 0; ic < ICPI; ic++)
                        {
#pragma HLS PIPELINE II = 1
                            if (padding){
                                output_read = 0;
                            }
                            else{
                                output_read = line_buff[(h + ki) % BUFFER_ROWS][(r + kj) * ICPI + ic];
                            }
                            for (T_C pi = 0; pi < PI; pi++)
                            {
                                output_pack.data[pi] = output_read.range((pi + 1) * AW - 1, pi * AW);
                            }
                            act_out.write(output_pack);
                        }
                    }
                }
            }
        }
    }
}

template <int PI, int IC, int HEIGHT, int WIDTH, int AW>
void conv_3x3_line_buffer_stride2_fifo_serial(
    hls::stream<BundleT<PI, ap_int<AW>>> &act_in,
    hls::stream<BundleT<PI, ap_int<AW>>> &act_out
)
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



    ap_int<PI * AW> line_buff[BUFFER_ROWS][BUFFER_WIDTH];
    ap_int<PI * AW> output_read = 0;
    BundleT<PI, ap_int<AW>> output_pack;
  
    for (int r = 0; r < BUFFER_ROWS; r++)
    {
        for (int h = 0; h < BUFFER_WIDTH; h++)
        {
#pragma HLS PIPELINE
            line_buff[r][h] = 0;
        }
    }

    int read_count = 0;
    int in_x = 0, in_y = 0;
    int out_x = 0, out_y = 0;
    for(T_HW h = -1; h < HEIGHT; h++){
        for(T_HW r = -1; r < WIDTH; r++){
            bool out_enable = (h >= 0) && (r >= 0) && (h % 2 == 0) && (r % 2 == 0);
            bool read_enable = read_count < HEIGHT * WIDTH;
            if (read_enable){
                for(T_C ic = 0; ic < ICPI; ic++){
#pragma HLS PIPELINE II = 1
                    BundleT<PI, ap_int<AW>> data_in = act_in.read();
                    ap_int<PI *AW> data_repacked = 0;
                    for (T_C i = 0; i < PI; i++)
                    {
                        data_repacked((i + 1) * AW - 1, i * AW) = data_in.data[i];
                    }
                    line_buff[in_y % BUFFER_ROWS][in_x * ICPI + ic] = data_repacked;
                }
                in_x++;
                if (in_x == WIDTH){
                    in_x = 0;
                    in_y++;
                }
                read_count++;
            }
            if (out_enable){
                for (ap_int<4> ki = -1; ki < 2; ki++)
                { // row
                    for (ap_int<4> kj = -1; kj < 2; kj++)
                    { // col
                        bool padding = h + ki < 0 || h + ki >= HEIGHT || r + kj < 0 || r + kj >= WIDTH;
                        for (T_C ic = 0; ic < ICPI; ic++)
                        {
#pragma HLS PIPELINE II = 1
                            if (padding){
                                output_read = 0;
                            }
                            else{
                                output_read = line_buff[(h + ki) % BUFFER_ROWS][(r + kj) * ICPI + ic];
                            }
                            for (T_C pi = 0; pi < PI; pi++)
                            {
                                output_pack.data[pi] = output_read.range((pi + 1) * AW - 1, pi * AW);
                            }
                            act_out.write(output_pack);
                        }
                    }
                }
            }
        }
    }


}

template <int PI, int IC, int HEIGHT, int WIDTH, int AW>
void conv_3x3_line_buffer_first_layer(
    hls::stream<BundleT<PI, ap_int<AW>>> &act_in,
    hls::stream<BundleT<9, ap_int<PI * AW>>> &act_out,
    hls::stream<T_K> &token_in, hls::stream<T_K> &token_out,
    hls::stream<T_K> &token_stride2)
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

    bool token_read_enable = 1; // enable the first read
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
#pragma HLS ARRAY_PARTITION variable = line_buff complete dim = 1

    ap_uint<WIDTH> valid[BUFFER_ROWS];
#pragma HLS ARRAY_PARTITION variable = valid complete dim = 1

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
            // stride 2 of tensor coordinate
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
        token_read_enable =
            data_read_enable; // if data read in current iteration, then can
                              // start to read token in next iteration

        // cout<<"rep"<<rep<<"\t";
        // cout<<"new_token.x:"<<new_token.x<<" new_token.y:"<<new_token.y<<"
        // new_token.end:"<<new_token.end<<endl; cout<<"key_even:"<<key_even<<"
        // key_odd:"<<key_odd<<endl; cout<<"pop_even:"<<pop_even<<"
        // pop_odd:"<<pop_odd<<endl;
        // cout<<"olddest_even_token_reg.x:"<<olddest_even_token_reg.x<<"
        // olddest_even_token_reg.y:"<<olddest_even_token_reg.y<<endl;
        // cout<<"olddest_odd_token_reg.x:"<<olddest_odd_token_reg.x<<"
        // olddest_odd_token_reg.y:"<<olddest_odd_token_reg.y<<endl;

        // cout<<"lastest_token.x:"<<lastest_token.x<<"
        // lastest_token.y:"<<lastest_token.y<<"
        // lastest_token.end:"<<lastest_token.end<<endl;
        // cout<<"oldest_token.x:"<<oldest_token.x<<"
        // oldest_token.y:"<<oldest_token.y<<"
        // oldest_token.end:"<<oldest_token.end<<endl;
        // cout<<"output_valid_one_line:"<<out_valid_one_line<<"
        // output_valid_multi_line:"<<out_valid_multi_line<<"
        // output_valid:"<<output_valid<<endl; cout<<"
        // data_read_enable:"<<data_read_enable<<"
        // token_read_enable:"<<token_read_enable<<endl;

        // update bitmap
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
            BundleT<PI, ap_int<AW>> data_in = act_in.read();
            ap_int<PI *AW> data_repacked = 0;
            for (T_C i = 0; i < PI; i++)
            {
#pragma HLS UNROLL
                data_repacked((i + 1) * AW - 1, i * AW) = data_in.data[i];
            }
            line_buff[lastest_token.y % BUFFER_ROWS][lastest_token.x] =
                data_repacked;
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
                    // bool valid_point = not_padding && valid[oldest_token.y +
                    // ki][oldest_token.x + kj]; cout<<valid_point<<" ";
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
                // cout<<endl;
            }
            act_out.write(win);
        }

        // cout<<endl;
    }
}