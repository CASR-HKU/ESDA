#include "top.h"
//
#include "tb_func.h"

int main() {
    // declare buffer
    static ap_int<CFG_AW * CFG_TOP_PIC> act_in_buffer[65536];
    static ap_int<CFG_AW * CFG_TOP_POC> act_out_buffer[65536];
    static ap_int<CFG_MW> mask_buffer[65536];

    int num_nz = CFG_TOP_IH * CFG_TOP_IW;
    // num_nz = read_tb_mask<1, CFG_MW>(mask_buffer, CFG_TOP_IH * CFG_TOP_IW,
    //                                  "tb_spatial_mask.txt");

    int input_feature_size = num_nz * CFG_TOP_IC;

    // readfile<CFG_AW, CFG_TOP_PIC>(act_in_buffer, input_feature_size,
    //                               "tb_input_feature.txt");

    // print mask buffer
    // int rep = ceil_div<CFG_MW>(CFG_TOP_IH * CFG_TOP_IW);
    // for (int i = 0; i < rep; i++) {
    //     ap_int<CFG_MW> tmp = mask_buffer[i];
    //     for (int j = 0; j < CFG_MW; j++) {
    //         cout << tmp[j] << " ";
    //         if (j % CFG_TOP_IW == CFG_TOP_IW - 1) cout << endl;
    //     }
    //     cout << endl;
    // }

    cout << "-------------start testing conv input--------------" << endl;

    top(act_in_buffer, act_out_buffer, mask_buffer, num_nz);

    // int output_feature_size = 0;
    // output_feature_size = count_number_of_lines("tb_output.txt");

    // FILE *f_gt;
    // f_gt = fopen("tb_output.txt", "r");

    // int count = 0;
    // int err_cnt = 0;

    // for (int i = 0; i < output_feature_size / CFG_TOP_POC; i++) {
    //     ap_uint<CFG_AW *CFG_TOP_POC> rd = act_out_buffer[i];
    //     for (int j = 0; j < CFG_TOP_POC; j++) {
    //         int out_read =
    //             (ap_int<CFG_AW>)(rd.range((j + 1) * CFG_AW - 1, j * CFG_AW));
    //         int tmp;
    //         fscanf(f_gt, "%d", &tmp);
    //         if (tmp != out_read) {
    //             cout << "failed at:" << count << "\tgt:" << tmp
    //                  << "\tout:" << out_read << endl;
    //             err_cnt++;
    //         } else {
    //             cout << "passed at:" << count << "\tgt:" << tmp
    //                  << "\tout:" << out_read << endl;
    //         }
    //         count++;
    //     }
    // }

    // if (err_cnt == 0)
    //     cout << "Passed dense tb!" << endl;
    // else
    //     cout << "Failed with " << err_cnt << " errors!" << endl;
    // fclose(f_gt);

    // return (err_cnt == 0 ? 0 : 1);
    return 0;
}