python eventnet.py \
    --model_path /vol/datastore/eventNetModel/0801_Rosh \
    --results_path /vol/datastore/eventNetConfig/baseline_0727_new_runs \
    --hw zcu102_40res \
    --model Roshambo_real1 \
    --config 0

python eventnet.py \
    --model_path /vol/datastore/eventNetModel/0725_baselines \
    --results_path /vol/datastore/eventNetConfig/baseline_0725_runs \
    --hw zcu102_60res \
    --model DVS_0p5 \
    --config 0

python eventnet.py \
    --model_path /vol/datastore/eventNetModel/0801_Rosh \
    --results_path /vol/datastore/eventNetConfig/baseline_0727_new_runs \
    --model Roshambo_2 \
    --hw zcu102_40res \
    --config 1

python eventnet.py \
    --model_path /vol/datastore/eventNetModel/0801_Rosh \
    --results_path /vol/datastore/eventNetConfig/baseline_0727_new_runs \
    --model Roshambo_2 \
    --hw zcu102_60res \
    --config 1
