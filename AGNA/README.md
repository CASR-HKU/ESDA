# AGNA for ESDA

This branch contains modified AGNA code for ESDA.

## Environment

The installation of SCIPOPT suite can be complicated. Please directly use the SCIPOPT suite installed on `americano02` for now. To set up the environment, run the following command or add it to your `~/.bashrc` file.

```bash
export SCIPOPTDIR=/media/ssd512g4/scipoptsuite-8.0.3
```

Then build environment by running:

```bash
conda create -n esda-dse python
conda activate esda-dse
pip install gpkit pyscipopt
```

## Usage

**Important:** Make sure `$SCIPOPTDIR` is set up correctly.

Start with an example:

```bash
python eventnet.py \
    --model_path /vol/datastore/eventNetModel/0727_NASModel \
    --model_name DVS_1890 \
    --hw_path /vol/datastore/eventNetHWConfig \
    --hw_name zcu102_80res \
    --results_path /vol/datastore/eventNetDSE
```

If you see output like this, then your environment is set up correctly:

```bash
[22:02:03]-INFO-root Result saved to /vol/datastore/eventNetDSE/DVS_1890-zcu102_80res/en-result.json
```

Let's go through all the basic arguments in `eventnet.py`:

```bash
python eventnet.py \
    --model_path <MODEL_PATH> \
    --model_name <MODEL_NAME> \
    --hw_path <HW_PATH> \
    --hw_name <HW_NAME> \
    --results_path <RESULTS_PATH>
```

With the above command, the script will load the model config from `<MODEL_PATH>/<MODEL_NAME>/model.json`, and the hardware config from `<HW_PATH>/<HW_NAME>.json`.

The results and the log will be saved to `<RESULTS_PATH>/<MODEL_NAME>-<HW_NAME>`. The results will be saved as `en-result.json` in the folder. All other files are for debugging purposes.

Most of the arguments have default values. You can find them in `eventnet.py`.
