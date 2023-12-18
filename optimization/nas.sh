make -f eventnetblock.mk nas -j4 -k NAS_NAME=DVS_new
make -f eventnetblock.mk nas -j4 -k NAS_NAME=ASL_fixed
make -f eventnetblock.mk nas -j4 -k NAS_NAME=NCal

make -f eventnetblock.mk nas -j4 -k \
    NAS_NAME=DVS \
    NAS_MODEL_PATH=/vol/datastore/eventNetModel/0731/DVS

make -f eventnetblock.mk nas -j4 -k NAS_NAME=NMNIST
make -f eventnetblock.mk nas -j4 -k NAS_NAME=Roshambo
