.PHONY: test all gen try_until_fail ip_all hw_all
.PHONY: hls_csim hls_cosim hls_export run_hls extract_ip
.PHONY: run_vivado extract_hw evaluate_hw
.PHONY: clean clean_prj clean_vivado clean_hls

ifeq ($(origin Q), undefined)
MYOUT := /dev/stdout
SCPFLAGS := 
else
MYOUT := /dev/null
SCPFLAGS := -q
endif

HLS_FLOW ?= csim
HLS_PRJ_NAME ?= sp_hls_new_proj
HLS_SRC_DIR ?= ..

VIVADO_PRJ_NAME ?= sp_new_vivado_proj

HW_NAME ?= $(notdir ${CURDIR})
HW_HOST := zcu102
HW_MAINDIR := /home/xilinx/jupyter_notebooks/event_spconv
HW_DIR := ${HW_MAINDIR}/${HW_NAME}
EVAL_TARGET ?= single_run

test:
	@sleep 1
	@RET=$(shell expr $${RANDOM} % 2) && \
		echo "Current project: ${CURDIR} return $${RET}" && \
		exit $${RET}

all:
	$(MAKE) ip_all
	$(MAKE) hw_all

gen:
# exit with `python` exit code instead of `tee`
	@python gen_code.py -f 2>&1 | tee gen_code.log > ${MYOUT}; exit $${PIPESTATUS[0]}
	@python gen_data.py 2>&1 | tee gen_data.log > ${MYOUT}; exit $${PIPESTATUS[0]} 

try_until_fail:
	@for i in $$(seq 1 25); do \
		echo "try $$i"; \
		$(MAKE) gen hls_csim Q= || exit 1; \
	done

ip_all:
	$(MAKE) hls_export
	$(MAKE) extract_ip

hw_all:
	$(MAKE) run_vivado
	$(MAKE) extract_hw

hls_csim:
# with 5 min timeout
	@timeout 5m $(MAKE) run_hls HLS_FLOW=csim

hls_cosim:
	$(MAKE) run_hls HLS_FLOW=cosim

hls_export:
	$(MAKE) run_hls HLS_FLOW=export
	@echo "IP exported to prj/${HLS_PRJ_NAME}/export.zip"

hls_vsynth:
	$(MAKE) run_hls HLS_FLOW=vsynth

run_hls:
	@echo "set HLS_FLOW ${HLS_FLOW}" > prj/hls_conf.tcl
	@echo "set HLS_PRJ_NAME ${HLS_PRJ_NAME}" >> prj/hls_conf.tcl
	@echo "set HLS_SRC_DIR ${HLS_SRC_DIR}" >> prj/hls_conf.tcl
	@test -v XILINX_HLS || (echo "ERROR: XILINX_HLS not exists" && exit 1)
	@echo "#include \"$${XILINX_HLS}/include/gmp.h\"" > fixgmp.h
	@echo "#include \"$${XILINX_HLS}/include/mpfr.h\"" >> fixgmp.h
	@echo "runnning HLS with ${HLS_FLOW}" flow
	@cd prj; \
		mkdir -p log; \
		vitis_hls hls.tcl -l log/vitis_hls.log > ${MYOUT}

extract_ip:
	@test -e prj/${HLS_PRJ_NAME}/export.zip || \
		(echo "ERROR: export.zip not found" && exit 1)
	@rm -rf prj/ip_repo && unzip -q prj/${HLS_PRJ_NAME}/export.zip -d prj/ip_repo
	@echo "IP extracted to prj/ip_repo"

run_vivado:
	@cd prj; \
		mkdir -p log; \
		vivado -mode batch -source vivado.tcl -journal log/vivado.jou -log log/vivado.log > ${MYOUT}

extract_hw:
	@mkdir -p hw
	@test -e prj/${VIVADO_PRJ_NAME}.xsa || \
		(echo "ERROR: ${VIVADO_PRJ_NAME}.xsa not found" && exit 1)
	@unzip -q -o prj/${VIVADO_PRJ_NAME}.xsa design_1.hwh -d hw/
	@mv hw/design_1.hwh hw/top.hwh
	@unzip -q -o prj/${VIVADO_PRJ_NAME}.xsa ${VIVADO_PRJ_NAME}.bit -d hw/
	@mv hw/${VIVADO_PRJ_NAME}.bit hw/top.bit
	@echo "HW extracted to hw/top.hwh and hw/top.bit"

evaluate_hw:
	@ssh ${HW_HOST} "mkdir -p ${HW_DIR}"
	@scp ${SCPFLAGS} hw/top.hwh ${HW_HOST}:${HW_DIR}/
	@scp ${SCPFLAGS} hw/top.bit ${HW_HOST}:${HW_DIR}/
	@scp ${SCPFLAGS} cfg.json ${HW_HOST}:${HW_DIR}/
	@scp ${SCPFLAGS} data/tb_input_feature.txt ${HW_HOST}:${HW_DIR}/
	@scp ${SCPFLAGS} data/tb_spatial_mask.txt ${HW_HOST}:${HW_DIR}/
	@scp ${SCPFLAGS} data/tb_output.txt ${HW_HOST}:${HW_DIR}/
	@echo "Running HW on ${HW_HOST}..."
	@ssh root@${HW_HOST} "cd ${HW_DIR}; make -f ../evaluate.mk ${EVAL_TARGET}"
	@scp ${SCPFLAGS} ${HW_HOST}:${HW_DIR}/evaluate.log hw/
	@head -n 3 hw/evaluate.log
	@echo "HW evaluated and results saved to hw/evaluate.log"
	
clean:
	@echo "Not implemented yet" && exit 1

clean_prj:
	@echo "This will delete all files in prj/ except *.tcl."
	@echo "Press [Enter] to continue, or Ctrl+C to cancel..."
	@read
	@find ./prj -mindepth 1 -maxdepth 1 -not -name '.gitignore' -not -name '*.tcl' -exec rm -rf {} \;

clean_vivado:
	@rm -rf prj/${VIVADO_PRJ_NAME}

clean_hls:
	@rm -rf prj/${HLS_PRJ_NAME}
