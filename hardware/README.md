# Workflow Automation

## Preparation

1. Copy template to `<target>`:
    ```bash
    cp -r template <target>
    ```

1. Go to `<target>`:
    ```bash
    cd <target>  # or
    code -n <target>  # to open in vscode
    ```

1. Modify `cfg.json`.

1. Generate `top.cpp`, `para.h` and `weight.h`:
    ```bash
    make gen
    ```

## `vitis_hls`

- Run `vitis_hls` and extract ip to `prj/ip_repo/`:
    ```bash
    make ip_all
    ```
- Run `csim` only:
    ```bash
    make hls_csim
    ```
- Check `prj/log/vitis_hls.log`:
    ```bash
    tail prj/log/vitis_hls.log  # use -f to follow
    ```
- Check project in GUI:
    ```bash
    cd prj
    vitis_hls -p <HLS_PRJ_NAME>
    ```

## `vivado`

- Run `vivado` and extract hw to `hw/`:
    ```bash
    make hw_all
    ```
- Check `prj/log/vivado.log`:
    ```bash
    tail prj/log/vivado.log  # use -f to follow
    ```
- Check project in GUI:
    ```bash
    cd prj
    vivado <VIVADO_PRJ_NAME>.xpr
    ```
## Hardware evaluation

- Add ssh key to `xilinx` and `root` of the board.

- Run:
    ```bash
    make evaluate_hw
    ```

- Check `hw/evaluate.log`:
    ```bash
    tail hw/evaluate.log
    ```
    