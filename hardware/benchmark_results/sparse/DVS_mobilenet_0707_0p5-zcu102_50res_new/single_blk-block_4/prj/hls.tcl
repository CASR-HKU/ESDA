# load hls_conf.tcl
if {[file exists hls_conf.tcl]} {
    source hls_conf.tcl
} else {
    puts "WARNING: hls_conf.tcl not found, use default configurations"
    set HLS_FLOW csim
    set HLS_PRJ_NAME sp_hls_new_proj
    set HLS_SRC_DIR ..
}

# check HLS_FLOW
if {[string match "csim" ${HLS_FLOW}]} {
    set HLS_CMD_LIST "csim"
} elseif {[string match "cosim_only" ${HLS_FLOW}]} {
    set HLS_CMD_LIST "cosim"
} elseif {[string match "cosim" ${HLS_FLOW}]} {
    set HLS_CMD_LIST "csim csynth cosim"
} elseif {[string match "export" ${HLS_FLOW}]} {
    set HLS_CMD_LIST "csim csynth cosim export"
} elseif {[string match "vsynth" ${HLS_FLOW}]} {
    set HLS_CMD_LIST "csynth vsynth"
} else {
    puts "WARNING: Unknown HLS_FLOW: ${HLS_FLOW}"
    set HLS_CMD_LIST ""
}

# create project
open_project ${HLS_PRJ_NAME}
set_top top

if {[info exists CFLAG_SRC] && ${CFLAG_SRC} != ""} {
    add_files ${HLS_SRC_DIR}/top.cpp -cflags ${CFLAG_SRC}
} else {
    add_files ${HLS_SRC_DIR}/top.cpp
}

add_files -tb ${HLS_SRC_DIR}/tb.cpp -cflags "-Wno-unknown-pragmas"
add_files -tb ${HLS_SRC_DIR}/data/tb_input_feature.txt
add_files -tb ${HLS_SRC_DIR}/data/tb_spatial_mask.txt
add_files -tb ${HLS_SRC_DIR}/data/tb_output.txt
add_files -tb ${HLS_SRC_DIR}/data/mask_0.1.txt
add_files -tb ${HLS_SRC_DIR}/data/mask_0.2.txt
add_files -tb ${HLS_SRC_DIR}/data/mask_0.3.txt
add_files -tb ${HLS_SRC_DIR}/data/mask_0.4.txt
add_files -tb ${HLS_SRC_DIR}/data/mask_0.5.txt
add_files -tb ${HLS_SRC_DIR}/data/mask_0.6.txt
add_files -tb ${HLS_SRC_DIR}/data/mask_0.7.txt
add_files -tb ${HLS_SRC_DIR}/data/mask_0.8.txt
add_files -tb ${HLS_SRC_DIR}/data/mask_0.9.txt
add_files -tb ${HLS_SRC_DIR}/data/mask_1.0.txt


open_solution "solution1" -flow_target vivado
set_part {xczu9eg-ffvb1156-2-e}
create_clock -period 3 -name default

# run HLS_CMD_LIST
if {[string match "*csim*" ${HLS_CMD_LIST}]} {
    csim_design
} else {
    puts "skip csim"
}
if {[string match "*csynth*" ${HLS_CMD_LIST}]} {
    csynth_design
} else {
    puts "skip csynth"
}
if {[string match "*cosim*" ${HLS_CMD_LIST}]} {
    cosim_design
} else {
    puts "skip cosim"
}
if {[string match "*export*" ${HLS_CMD_LIST}]} {
    export_design -format ip_catalog -output ./${HLS_PRJ_NAME}/
} else {
    puts "skip export"
}
if {[string match "*vsynth*" ${HLS_CMD_LIST}]} {
    export_design -flow syn -rtl verilog -format ip_catalog -output ./${HLS_PRJ_NAME}/
} else {
    puts "skip vsynth"
}
