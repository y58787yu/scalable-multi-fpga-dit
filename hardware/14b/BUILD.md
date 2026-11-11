# Cosmos-Predict2-14B

Build from the repository root with `python hardware/build.py --model 14b --output build/cosmos-14b`.

The four-port node design serves TP4×SP3. `compute/` contains the fixed 36-layer source and five-slice compute RTL. The export registers the MAC clock and packed attention memories. `board.tcl` assigns the same hardware ID to the links and the batch controller. This reproduces the controller identity correction from the deployed routed ECO at block-design construction time.

The production pair is the supplied four-port overlay and mode-2 runtime. Fresh place-and-route starts from the RTL and tool-generated IP.
