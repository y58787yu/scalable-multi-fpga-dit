# Cosmos-Predict2-2B

Build from the repository root with `python hardware/build.py --model 2b --output build/cosmos-2b`.

The single two-port node design serves both TP2×SP1 and TP2×SP2. `compute/` contains the fixed 28-layer source and the RTL for five 256-lane MAC slices. The export binds a 300 MHz MAC clock, registers the packed Q, probability and output-state memories, and uses the corrected sized-literal syntax of the deployed MAC leaf.

`board.tcl` connects the P1 reduction path and the P2 bulk path. `shell/locate_gt.tcl` assigns P1/P2 to channels X1Y13/X1Y14 of the shared GT quad after synthesis. Select the matching SP1 or SP2 runtime profile when deploying the generated overlay.
