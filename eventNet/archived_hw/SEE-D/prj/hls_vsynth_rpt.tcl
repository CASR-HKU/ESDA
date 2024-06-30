open_project -read_only sp_hls_new_proj/solution1/impl/verilog/project.xpr
open_run bd_0_hls_inst_0_synth_1
get_cells -hierarchical -filter {REF_NAME==design_1_top_0_0_top}

foreach i [get_cells -hierarchical -filter {PARENT=~inst/grp_wrapper_fu_*/conv_1x1_3x3_dw_1x1_* && REF_NAME=~RAMB36*}] {puts $i}
foreach i [get_cells -hierarchical -filter {PARENT=~inst/grp_wrapper_fu_*/conv_1x1_3x3_dw_1x1_* && REF_NAME=~RAMB18*}] {puts $i}