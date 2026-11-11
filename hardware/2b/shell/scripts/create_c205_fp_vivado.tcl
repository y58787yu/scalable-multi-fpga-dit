# Vendor IEEE single-precision floating-point operators, separate rounding stages.
# Blocking AXIS flow control is essential: ready exists on BOTH operands/result.
foreach {name operation latency} {c205_fp32_add Add_Subtract 11 c205_fp32_mul Multiply 8} {
 create_ip -name floating_point -vendor xilinx.com -library ip -version 7.1 -module_name $name
 set_property -dict [list CONFIG.Operation_Type $operation \
  CONFIG.A_Precision_Type Single CONFIG.Result_Precision_Type Single \
  CONFIG.Flow_Control Blocking CONFIG.Has_ARESETn true \
  CONFIG.Maximum_Latency false CONFIG.C_Latency $latency CONFIG.C_Rate 1 \
  CONFIG.Has_RESULT_TREADY true] [get_ips $name]
 if {$operation eq "Add_Subtract"} {set_property CONFIG.Add_Sub_Value Add [get_ips $name]}
}
