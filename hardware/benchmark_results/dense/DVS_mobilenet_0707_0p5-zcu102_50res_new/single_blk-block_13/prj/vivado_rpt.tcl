open_project -read_only sp_new_vivado_proj/sp_new_vivado_proj.xpr
open_run design_1_top_0_0_synth_1
get_cells -hierarchical -filter {REF_NAME==design_1_top_0_0_top}