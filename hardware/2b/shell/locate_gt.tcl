# The GT primitives become visible after synthesis has linked the OOC IPs.
foreach {key primitive scope expected_site} {
 p1_gt_channel GTHE4_CHANNEL *links/ports?1?*phy/* GTHE4_CHANNEL_X1Y13
 p2_gt_channel GTHE4_CHANNEL *links/ports?2?*phy/* GTHE4_CHANNEL_X1Y14
 shared_gt_common GTHE4_COMMON *links/ports?1?*phy/* GTHE4_COMMON_X1Y3
} {
 set targets [get_cells -hier -quiet -filter "REF_NAME == $primitive && NAME =~ $scope"]
 if {[llength $targets]!=1} {error "Two-port $key requires one actual mapped primitive: $targets"}
 set_property LOC $expected_site $targets
 set observed_site [get_property LOC $targets]
 if {$observed_site ne $expected_site} {error "Two-port $key LOC was not applied"}
 dict set two_port_gt_locations ${key}_cell [lindex $targets 0]
 dict set two_port_gt_locations ${key}_loc $observed_site
}
