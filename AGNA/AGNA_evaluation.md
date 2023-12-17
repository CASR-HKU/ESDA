
python eventnet.py \
    --model_path /vol/datastore/baoheng/eventModel/bit_bias16 --hw_path /vol/datastore/EDSA/eventNetHWConfig \
    --model_name DVS_1890_shift16 \
    --hw_name zcu102_80res \
    --results_path /vol/datastore/EDSA/AE_test/eventDSE

python eventnet.py \
    --model_path /vol/datastore/baoheng/eventModel/bit_bias16 --hw_path /vol/datastore/EDSA/eventNetHWConfig \
    --model_name DVS_0p5_shift16 \
    --hw_name zcu102_60res \
    --results_path /vol/datastore/EDSA/AE_test/eventDSE

python eventnet.py \
    --model_path /vol/datastore/baoheng/eventModel/bit_bias16 --hw_path /vol/datastore/EDSA/eventNetHWConfig \
    --model_name NMNIST_shift16 \
    --hw_name zcu102_60res \
    --results_path /vol/datastore/EDSA/AE_test/eventDSE

python eventnet.py \
    --model_path /vol/datastore/baoheng/eventModel/bit_bias16 --hw_path /vol/datastore/EDSA/eventNetHWConfig \
    --model_name ASL_0p5_shift16 \
    --hw_name zcu102_80res \
    --results_path /vol/datastore/EDSA/AE_test/eventDSE

python eventnet.py \
    --model_path /vol/datastore/baoheng/eventModel/bit_bias16 --hw_path /vol/datastore/EDSA/eventNetHWConfig \
    --model_name ASL_2929_shift16 \
    --hw_name zcu102_80res \
    --results_path /vol/datastore/EDSA/AE_test/eventDSE

 python eventnet.py \
    --model_path /vol/datastore/baoheng/eventModel/bit_bias16 --hw_path /vol/datastore/EDSA/eventNetHWConfig \
    --model_name Roshambo_shift16_2 \
    --hw_name zcu102_80res \
    --results_path /vol/datastore/EDSA/AE_test/eventDSE

python eventnet.py \
    --model_path /vol/datastore/baoheng/eventModel/bit_bias16 --hw_path /vol/datastore/EDSA/eventNetHWConfig \
    --model_name NCal_2751_shift32_3 \
    --hw_name zcu102_80res \
    --results_path /vol/datastore/EDSA/AE_test/eventDSE

python eventnet.py \
    --model_path /vol/datastore/baoheng/eventModel/bit_bias16 --hw_path /vol/datastore/EDSA/eventNetHWConfig \
    --model_name NCal_w0p5_shift32_2 \
    --hw_name zcu102_50res \
    --results_path /vol/datastore/EDSA/AE_test/eventDSE


