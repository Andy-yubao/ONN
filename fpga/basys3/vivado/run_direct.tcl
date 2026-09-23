# Single-process non-project flow for hosts whose Vivado project run launcher fails.
set root [file normalize [file join [file dirname [info script]] .. .. ..]]
set report_dir [file join $root fpga basys3 reports]
file mkdir $report_dir
set stage [lindex $argv 0]
if {$stage eq "synth"} {
    set include_dir [file join $root model snn export]
    foreach name {onn_snn_params_pkg.sv onn_signed_saturate.sv onn_if_unit.sv onn_sparse_conv1_if1.sv onn_sparse_conv2_if2.sv onn_sparse_readout.sv onn_snn_core_sparse.sv} {
        read_verilog -sv [file join $root fpga rtl $name]
    }
    foreach name {onn_uart_rx.sv onn_uart_tx.sv onn_basys3_top.sv} {
        read_verilog -sv [file join $root fpga basys3 rtl $name]
    }
    set_property include_dirs [list $include_dir] [current_fileset]
    read_xdc [file join $root fpga basys3 basys3.xdc]
    synth_design -top onn_basys3_top -part xc7a35tcpg236-1 -flatten_hierarchy none -directive RuntimeOptimized
    report_utilization -hierarchical -file [file join $report_dir synthesis_utilization.rpt]
    report_timing_summary -file [file join $report_dir synthesis_timing.rpt]
    report_drc -file [file join $report_dir synthesis_drc.rpt]
    write_checkpoint -force [file join $report_dir synthesis.dcp]
} elseif {$stage eq "impl"} {
    open_checkpoint [file join $report_dir synthesis.dcp]
    opt_design
    place_design
    phys_opt_design
    route_design
    report_utilization -hierarchical -file [file join $report_dir implementation_utilization.rpt]
    report_timing_summary -file [file join $report_dir implementation_timing.rpt]
    report_drc -file [file join $report_dir implementation_drc.rpt]
    write_checkpoint -force [file join $report_dir implementation.dcp]
    set setup_paths [get_timing_paths -setup -max_paths 1]
    if {[llength $setup_paths] == 0} {
        error "No routed setup timing path was found; refusing to write a bitstream"
    }
    set worst_slack [get_property SLACK [lindex $setup_paths 0]]
    if {$worst_slack < 0.0} {
        error "Routed setup timing failed (WNS $worst_slack ns); refusing to write a bitstream"
    }
    write_bitstream -force [file join $report_dir onn_basys3.bit]
} else {
    error "Usage: vivado -mode batch -source run_direct.tcl -tclargs synth|impl"
}
puts "ONN Basys3 direct $stage completed; reports: $report_dir"
