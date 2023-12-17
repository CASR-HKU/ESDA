#include "top.h"
//
#include "tb_func.h"

int main() {
    // declare buffer
    static ap_uint<CFG_AW * CFG_TOP_PIC> act_in_buffer[65536];
    static ap_int<32> act_out_buffer[65536];
    static ap_int<CFG_MW> mask_buffer[65536];

    int num_nz = 0;
    num_nz = read_tb_mask<CFG_MW>(mask_buffer, CFG_TOP_IH, CFG_TOP_IW,
                                     "tb_spatial_mask.txt");

    int input_feature_size = num_nz * CFG_TOP_IC;

    readfile<CFG_AW, CFG_TOP_PIC>(act_in_buffer, input_feature_size,
                                  "tb_input_feature.txt");

    // print mask buffer
    int rep = ceil_div<CFG_MW>(CFG_TOP_IH * CFG_TOP_IW);
    for (int i = 0; i < rep; i++) {
        ap_int<CFG_MW> tmp = mask_buffer[i];
        for (int j = 0; j < CFG_MW; j++) {
            cout << tmp[j] << " ";
            if (j % CFG_TOP_IW == CFG_TOP_IW - 1) cout << endl;
        }
        cout << endl;
    }

    cout << "-------------start testing conv input--------------" << endl;

    top(act_in_buffer, act_out_buffer, mask_buffer, num_nz);

    std::ifstream inputFile("tb_output.txt");

    int err_cnt = 0;

    std::vector<int> intArray;
    std::string line;
    while (std::getline(inputFile, line)) {
        int number = std::stoi(line);
        intArray.push_back(number);
    }

    inputFile.close();

    for (int i = 0; i < intArray.size(); i++) {
            ap_int<32> rd = intArray[i];
            if (rd != act_out_buffer[i]) {
                err_cnt++;
        }
    }
    for (int i = 0; i < CFG_TOP_OC; i++) {
        ap_int<32> rd = act_out_buffer[i];
        cout << i << ":" << rd << endl;
    }


    if (err_cnt == 0)
        cout << "Passed dense tb!" << endl;
    else
        cout << "Failed with " << err_cnt << " errors!" << endl;

    return 0;
}