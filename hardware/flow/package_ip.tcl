if {$argc != 1} { error "package_ip.tcl COMPONENT_XML" }
set core [ipx::open_core -set_current false [file normalize [lindex $argv 0]]]
ipx::update_checksums $core
ipx::save_core $core
exit
