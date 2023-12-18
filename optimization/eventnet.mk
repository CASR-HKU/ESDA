MODEL_PATH := /vol/datastore/eventNetModel
RESULTS_PATH := /vol/datastore/eventNetConfig/default_runs
MODEL_LIST := DVS_mobilenet_0703 DVS_mobilenet_0707_0p75 DVS_mobilenet_0707_0p5
HW_LIST := zcu102_50res zcu102_60res zcu102_75res zcu102_80res zcu102_full
MODEL_HW_LIST := $(foreach model,$(MODEL_LIST),$(foreach hw,$(HW_LIST),$(model)-$(hw)))

NAS_NAME ?= DVS_new
NAS_MODEL_PATH := /vol/datastore/eventConfigSearching/model_json_0721/${NAS_NAME}
NAS_RESULTS_PATH := /vol/datastore/eventNetConfig/nas_${NAS_NAME}_runs
NAS_MODEL_LIST := $(shell find ${NAS_MODEL_PATH} -name "model_*.json" -printf "%f\n")
NAS_MODEL_LIST := $(basename $(NAS_MODEL_LIST))
NAS_HW_LIST := zcu102_80res
NAS_MODEL_HW_LIST := $(foreach model,$(NAS_MODEL_LIST),$(foreach hw,$(NAS_HW_LIST),nas-$(model)-$(hw)))

BM_MODEL_PATH := ${MODEL_PATH}
BM_RESULTS_PATH := /vol/datastore/eventNetConfig/benchmark_runs
BM_MODEL_LIST := DVS_mobilenet_0707_0p5
BM_HW_LIST := zcu102_50res
BM_MODEL_HW_LIST := $(foreach model,$(BM_MODEL_LIST),$(foreach hw,$(BM_HW_LIST),bm-$(model)-$(hw)))

E2E_MODEL_PATH := /vol/datastore/eventNetModel/DVS_mobilenet_0719
E2E_RESULTS_PATH := /vol/datastore/eventNetConfig/e2e_runs
E2E_MODEL_LIST := $(shell find ${E2E_MODEL_PATH} -type d -name int_*_shift32 -printf "%f\n")
E2E_MODEL_LIST := $(basename $(E2E_MODEL_LIST))
E2E_HW_LIST := zcu102_60res zcu102_80res
E2E_MODEL_HW_LIST := $(foreach model,$(E2E_MODEL_LIST),$(foreach hw,$(E2E_HW_LIST),e2e-$(model)-$(hw)))

BL_MODEL_PATH := /vol/datastore/eventNetModel/0725_baselines
BL_RESULTS_PATH := /vol/datastore/eventNetConfig/baseline_0725_runs
BL_MODEL_LIST := $(shell find ${BL_MODEL_PATH} -mindepth 1 -maxdepth 1 -type d -printf "%f\n")
BL_MODEL_LIST := $(basename $(BL_MODEL_LIST))
BL_HW_LIST := zcu102_60res zcu102_80res
BL_MODEL_HW_LIST := $(foreach model,$(BL_MODEL_LIST),$(foreach hw,$(BL_HW_LIST),bl-$(model)-$(hw)))

.PHONY: check
.PHONY: all ${MODEL_HW_LIST}
.PHONY: nas ${NAS_MODEL_HW_LIST}
.PHONY: bm ${BM_MODEL_HW_LIST}
.PHONY: e2e ${E2E_MODEL_HW_LIST}
.PHONY: bl ${BL_MODEL_HW_LIST}


check:
	@echo "MODEL x HW:"
	@echo "ALL: $(words ${MODEL_LIST}) x $(words ${HW_LIST}) = $(words ${MODEL_HW_LIST})"
	@echo "MODEL_PATH: ${MODEL_PATH}"
	@echo "RESULTS_PATH: ${RESULTS_PATH}"
	@echo "NAS: $(words ${NAS_MODEL_LIST}) x $(words ${NAS_HW_LIST}) = $(words ${NAS_MODEL_HW_LIST})"
	@echo "NAS_MODEL_PATH: ${NAS_MODEL_PATH}"
	@echo "NAS_RESULTS_PATH: ${NAS_RESULTS_PATH}"
	@echo "BM: $(words ${BM_MODEL_LIST}) x $(words ${BM_HW_LIST}) = $(words ${BM_MODEL_HW_LIST})"
	@echo "BM_MODEL_PATH: ${BM_MODEL_PATH}"
	@echo "BM_RESULTS_PATH: ${BM_RESULTS_PATH}"
	@echo "E2E: $(words ${E2E_MODEL_LIST}) x $(words ${E2E_HW_LIST}) = $(words ${E2E_MODEL_HW_LIST})"
	@echo "E2E_MODEL_PATH: ${E2E_MODEL_PATH}"
	@echo "E2E_RESULTS_PATH: ${E2E_RESULTS_PATH}"
	@echo "BL: $(words ${BL_MODEL_LIST}) x $(words ${BL_HW_LIST}) = $(words ${BL_MODEL_HW_LIST})"
	@echo "BL_MODEL_PATH: ${BL_MODEL_PATH}"
	@echo "BL_RESULTS_PATH: ${BL_RESULTS_PATH}"

all: ${MODEL_HW_LIST}

${MODEL_HW_LIST}:
	@if [ -f ${RESULTS_PATH}/$@/en-result.json ]; then \
		echo "skip $@"; \
	else \
		echo "run $@"; \
		python eventnet.py \
			--model $(word 1,$(subst -, ,$@)) \
			--hw $(word 2,$(subst -, ,$@)) \
			--model_path ${MODEL_PATH} \
			--results_path ${RESULTS_PATH} \
			; \
	fi

nas: ${NAS_MODEL_HW_LIST}

${NAS_MODEL_HW_LIST}:
	@if [ -f ${NAS_RESULTS_PATH}/$@/en-result.json ]; then \
		echo "skip $@"; \
	else \
		echo "run $@"; \
		python eventnet.py \
		--nas \
		--model $(word 2,$(subst -, ,$@)) \
		--hw $(word 3,$(subst -, ,$@)) \
		--model_path ${NAS_MODEL_PATH} \
		--results_path ${NAS_RESULTS_PATH} \
		; \
	fi

bm: ${BM_MODEL_HW_LIST}

${BM_MODEL_HW_LIST}:
	@if [ -f ${BM_RESULTS_PATH}/$@/en-result.json ]; then \
		echo "skip $@"; \
	else \
		echo "run $@"; \
		python eventnetblock.py \
		--model $(word 2,$(subst -, ,$@)) \
		--hw $(word 3,$(subst -, ,$@)) \
		--model_path ${BM_MODEL_PATH} \
		--results_path ${BM_RESULTS_PATH} \
		; \
	fi

e2e: ${E2E_MODEL_HW_LIST}

${E2E_MODEL_HW_LIST}:
	@if [ -f ${E2E_RESULTS_PATH}/$@/en-result.json ]; then \
		echo "skip $@"; \
	else \
		echo "run $@"; \
		python eventnetblock.py \
		--model $(word 2,$(subst -, ,$@)) \
		--hw $(word 3,$(subst -, ,$@)) \
		--model_path ${E2E_MODEL_PATH} \
		--results_path ${E2E_RESULTS_PATH} \
		; \
	fi

bl: ${BL_MODEL_HW_LIST}

${BL_MODEL_HW_LIST}:
	@if [ -f ${BL_RESULTS_PATH}/$@/en-result.json ]; then \
		echo "skip $@"; \
	else \
		echo "run $@"; \
		python eventnet.py \
		--model $(word 2,$(subst -, ,$@)) \
		--hw $(word 3,$(subst -, ,$@)) \
		--model_path ${BL_MODEL_PATH} \
		--results_path ${BL_RESULTS_PATH} \
		; \
	fi
