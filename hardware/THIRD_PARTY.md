# Third-party hardware components

The vendored Verilog and Vivado constraints under `ethernet/` come from Alex Forencich's [verilog-ethernet](https://github.com/alexforencich/verilog-ethernet) and [verilog-axis](https://github.com/alexforencich/verilog-axis) projects. The source snapshots for both overlays contain the same selected files. Their original copyright and permission notices accompany the code. The transceiver and Ethernet wrappers under each `shell/rtl/` also retain their upstream notices.

The build creates the Zynq UltraScale+ PS, GT Wizard, clocking, AXI DMA/interconnect, reset, FIFO and floating-point IP from the installed AMD Vivado catalog. Use the ZCU102 board files and the AMD tools described in [BUILD.md](BUILD.md). The HLS source uses the AMD HLS numeric/stream headers and the RTL instantiates AMD XPM and device primitives from that installation.
