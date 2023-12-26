SUB_DIR_LIST := $(shell find . -mindepth 3 -maxdepth 3 -not -name ".*" -type d )

.PHONY: all $(SUB_DIR_LIST) check

check:
	@echo $(SUB_DIR_LIST)

all: $(SUB_DIR_LIST)

$(SUB_DIR_LIST):
	@echo "make -C $@"
	@$(MAKE) -C $@ ip_all > /dev/null 2>&1
