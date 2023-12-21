legal = lambda name: name.replace(".", "_")
legal_low = lambda name: legal(name).lower()
legal_up = lambda name: legal(name).upper()
cfg_of = lambda name, cfg: f"CFG_{legal_up(name)}_{cfg}"
afifo_of = lambda name: f"a_{legal_low(name)}"
tfifo_of = lambda name: f"t_{legal_low(name)}"
mfifo_of = lambda name: f"m_{legal_low(name)}"
ts2fifo_of = lambda name: f"ts2_{legal_low(name)}"
wbuf_of = lambda name: f"w_{legal_low(name)}"
wfile_of = lambda name: f"{legal_low(name)}_w"
sbuf_of = lambda name: f"s_{legal_low(name)}"
sfile_of = lambda name: f"{legal_low(name)}_s"
ibuf_of = lambda name: f"i_{legal_low(name)}"
ifile_of = lambda name: f"{legal_low(name)}_i"

npy_of_conv = lambda name, post: f"{legal_low(name)}_{post}.npy"
npy_of_block = lambda name, i, post: f"{legal_low(name)}_conv{i}_{post}.npy"
npy_of = lambda layer, i, post: (
    npy_of_conv(layer["name"], post)
    if (layer["type"] == "conv" or layer["type"] == "linear")
    else npy_of_block(layer["name"], i, post)
    if layer["type"] == "block"
    else None
)
