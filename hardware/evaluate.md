
After logging in zcu102, go to the directory '/home/xilinx/jupyter_notebooks/ESDA_2024_FPGA/' by 

cd /home/xilinx/jupyter_notebooks/ESDA_2024_FPGA/

Then run the following commands to generate the results.


1. Latency and power consumption evaluation

python evaluate_e2e.py -1 -d hw/ASL_2929 --enable_pm
python evaluate_e2e.py -1 -d hw/ASL_w0p5 --enable_pm
python evaluate_e2e.py -1 -d hw/DVS_1890 --enable_pm
python evaluate_e2e.py -1 -d hw/DVS_w0p5 --enable_pm
python evaluate_e2e.py -1 -d hw/NMNIST --enable_pm
python evaluate_e2e.py -1 -d hw/Roshambo --enable_pm
python evaluate_e2e.py -1 -d hw/NCal_2751 --enable_pm
python evaluate_e2e.py -1 -d hw/NCal_w0p5 --enable_pm

The results will be stored in the folder `results/` respectively.


2. End-to-end evaluation

python hw_e2e.py 1 -d hw/ASL_2929
python hw_e2e.py 1 -d hw/ASL_w0p5
python hw_e2e.py 1 -d hw/DVS_1890
python hw_e2e.py 1 -d hw/DVS_w0p5
python hw_e2e.py 1 -d hw/NMNIST
python hw_e2e.py 1 -d hw/Roshambo
python hw_e2e.py 1 -d hw/NCal_2751
python hw_e2e.py 1 -d hw/NCal_w0p5

The e2e inference results will be generated the same as the software model e2e.




