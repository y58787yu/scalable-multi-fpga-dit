# IP12 exact asynchronous assertion/synchronous release endpoints only.
# XPM keeps its own scoped CDC constraints. No asynchronous clock-group blanket.
set c209_five [get_cells -hierarchical -quiet -filter {REF_NAME =~ *c209_five_mac_rtl || ORIG_REF_NAME =~ *c209_five_mac_rtl}]
if {[llength $c209_five]!=1} {error "Exactly one five-shard broker required before CDC exceptions"}
set c209_broker [get_property NAME [lindex $c209_five 0]]
if {![regexp {^[A-Za-z0-9_/]+$} $c209_broker]} {error "Review unexpected five-shard hierarchy spelling"}
set c209_start_sync [get_cells -hierarchical -quiet -regexp [format {^%s/start_sync_reg\[0\]$} $c209_broker]]
if {[llength $c209_start_sync]!=1} {error "Exactly one five-shard first-stage start synchronizer required"}
set c209_start_d [get_pins -quiet -of_objects $c209_start_sync -filter {REF_PIN_NAME == D}]
if {[llength $c209_start_d]!=1} {error "Missing exact first-stage start D"}
set c209_domain_reset [get_cells -hierarchical -quiet -regexp [format {^%s/(slow|fast)_reset_pipe_reg\[[0-3]\]$} $c209_broker]]
set c209_read_reset [get_cells -hierarchical -quiet -regexp [format {^%s/shard_cdc\[[0-4]\]\.result_cdc/read_reset_pipe_reg\[[0-2]\]$} $c209_broker]]
if {[llength $c209_domain_reset]!=8 || [llength $c209_read_reset]!=15} {error "Five-shard reset inventory must be 8 domain plus 15 result registers"}
set c209_reset_pipe [concat $c209_domain_reset $c209_read_reset]
if {[llength [lsort -unique $c209_reset_pipe]]!=23} {error "Expected exactly 23 distinct asynchronous reset registers"}
foreach cell [concat $c209_start_sync $c209_reset_pipe] {
 if {[get_property ASYNC_REG $cell] ni {1 TRUE}} {error "Exact synchronization register lacks ASYNC_REG: $cell"}
}
set c209_reset_pins [get_pins -quiet -of_objects $c209_reset_pipe -filter {REF_PIN_NAME == CLR || REF_PIN_NAME == PRE}]
if {[llength $c209_reset_pins]!=23} {error "Incomplete exact reset-pin coverage"}
set_false_path -to $c209_start_d
set_false_path -to $c209_reset_pins
