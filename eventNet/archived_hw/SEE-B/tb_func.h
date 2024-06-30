// combine scale and bias
template <int SCALEW, int BIASW>
void read_scale_bias(ap_int<SCALEW + BIASW> *mem, int length,
                     const char *scale_name, const char *bias_name) {
    int count = 0;
    FILE *f_s, *f_b;
    f_s = fopen(scale_name, "r");
    f_b = fopen(bias_name, "r");
    int rep = (length);
    for (int i = 0; i < rep; i++) {
        ap_int<SCALEW + BIASW> temp;
        int tmp_s, tmp_b;
        fscanf(f_s, "%d", &tmp_s);
        fscanf(f_b, "%d", &tmp_b);
        temp.range(BIASW - 1, 0) = tmp_b;
        temp.range(SCALEW + BIASW - 1, BIASW) = tmp_s;
        mem[count++] = temp;
        //   cout<<temp<<endl;
    }
    fclose(f_s);
    fclose(f_b);
}

template <int W, int PA, typename T_OUT>
void readfile(T_OUT *mem, int length, const char *name) {
    int count = 0;
    FILE *f_s;
    f_s = fopen(name, "r");
    int rep = ceil_div<PA>(length);
    for (int i = 0; i < rep; i++) {
        T_OUT temp;
        for (int j = 0; j < PA; j++) {
            int tmp;
            fscanf(f_s, "%d", &tmp);
            //   cout<<"tmp:"<<tmp<<endl;
            temp.range(W * (j + 1) - 1, W * j) = tmp;
        }
        mem[count++] = temp;
        //   cout<<temp<<endl;
    }
    fclose(f_s);
}


template <int PA>
int read_tb_mask(ap_int<PA> *mem, int height, int width, const char *name) {
    int count = 0;
    FILE *f_s;
    f_s = fopen(name, "r");
    int nz_count = 0;
    int FLAG = 0;

    int width_div_round = ((width + PA - 1) / PA); 
    


    for (int h = 0; h < height; h++) {
        int width_count = 0;
        for(int w = 0; w < width_div_round; w++){
            ap_int<PA>  out_pack = 0;
            for(int i = 0; i < PA; i++){
                if(width_count >= width) break;
                int tmp;
                FLAG = fscanf(f_s, "%d", &tmp);
                if (FLAG == EOF) {
                    break;
                }
                out_pack[i] = tmp;
                if (tmp != 0) nz_count++;
                width_count++;
            }
            mem[count++] = out_pack;
        }
    }
    fclose(f_s);
    return nz_count;
}


template <int W, int PA, typename T_OUT>
int read_file_static(T_OUT *mem, int length, const char *name,
                     int start_index) {
    int count = start_index;
    FILE *f_s;
    f_s = fopen(name, "r");
    int rep = ceil_div<PA>(length);
    for (int i = 0; i < rep; i++) {
        T_OUT temp;
        for (int j = 0; j < PA; j++) {
            int tmp;
            fscanf(f_s, "%d", &tmp);
            temp.range(W * (j + 1) - 1, W * j) = tmp;
        }
        mem[count++] = temp;
        //   cout<<temp<<endl;
    }
    fclose(f_s);
    return count;
}

template <typename T_OUT>
T_OUT read_cprune(const char *name, int N) {
    int count = 0;
    FILE *f_s;
    f_s = fopen(name, "r");
    T_OUT cprune = 0;

    for (int i = 0; i < N; i++) {
        int tmp;
        fscanf(f_s, "%d", &tmp);
        cprune[i] = tmp;
    }
    fclose(f_s);
    return cprune;
}

int count_number_of_lines(const char *name) {
    int count = 0;
    FILE *f_s;
    f_s = fopen(name, "r");
    int tmp;
    while (fscanf(f_s, "%d", &tmp) != EOF) {
        count++;
    }
    fclose(f_s);
    return count;
}