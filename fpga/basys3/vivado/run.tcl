# Reproducible Vivado project build. Run from any directory with -tclargs synth|impl.
set root [file normalize [file join [file dirname [info script]] .. .. ..]]
set project_dir [file join $root fpga basys3 .vivado]
set report_dir [file join $root fpga basys3 reports]
file mkdir $report_dir
set stage [lindex $argv 0]
if {$stage eq "synth"} {
    create_project onn_basys3 $project_dir -part xc7a35tcpg236-1 -force
    set files [list \
        [file join $root fpga rtl onn_snn_params_pkg.sv] \
        [file join $root fpga rtl onn_signed_saturate.sv] \
        [file join $root fpga rtl onn_if_unit.sv] \
        [file join $root fpga rtl onn_sparse_conv1_if1.sv] \
        [file join $root fpga rtl onn_sparse_conv2_if2.sv] \
        [file join $root fpga rtl onn_sparse_readout.sv] \
        [file join $root fpga rtl onn_snn_core_sparse.sv] \
        [file join $root fpga basys3 rtl onn_uart_rx.sv] \
        [file join $root fpga basys3 rtl onn_uart_tx.sv] \
        [file join $root fpga basys3 rtl onn_basys3_top.sv]]
    add_files -fileset sources_1 -norecurse $files
    foreach name {conv1_weight.mem conv2_weight.mem readout_weight.mem} {
        add_files -fileset sources_1 -norecurse [file join $root model snn export $name]
    }
    set_property include_dirs [list [file join $root model snn export]] [get_filesets sources_1]
    add_files -fileset constrs_1 -norecurse [file join $root fpga basys3 basys3.xdc]
    set_property top onn_basys3_top [get_filesets sources_1]
    set_property STEPS.SYNTH_DESIGN.ARGS.FLATTEN_HIERARCHY none [get_runs synth_1]
    update_compile_order -fileset sources_1
    launch_runs synth_1 -jobs 4
    wait_on_run synth_1
    if {[get_property STATUS [get_runs synth_1]] ne "synth_design Complete!"} {
        error "Synthesis did not complete: [get_property STATUS [get_runs synth_1]]"
    }
    open_run synth_1
    report_utilization -hierarchical -file [file join $report_dir synthesis_utilization.rpt]
    report_timing_summary -file [file join $report_dir synthesis_timing.rpt]
    report_drc -file [file join $report_dir synthesis_drc.rpt]
    write_checkpoint -force [file join $report_dir synthesis.dcp]
} elseif {$stage eq "impl"} {
    open_project [file join $project_dir onn_basys3.xpr]
    if {[get_property STATUS [get_runs synth_1]] ne "synth_design Complete!"} {
        error "Run synthesis first"
    }
    launch_runs impl_1 -to_step route_design -jobs 4
    wait_on_run impl_1
    if {[get_property STATUS [get_runs impl_1]] ne "route_design Complete!"} {
        error "Implementation did not complete: [get_property STATUS [get_runs impl_1]]"
    }
    open_run impl_1
    report_utilization -hierarchical -file [file join $report_dir implementation_utilization.rpt]
    report_timing_summary -file [file join $report_dir implementation_timing.rpt]
    report_drc -file [file join $report_dir implementation_drc.rpt]
    write_checkpoint -force [file join $report_dir implementation.dcp]
} else {
    error "Usage: vivado -mode batch -source fpga/basys3/vivado/run.tcl -tclargs synth|impl"
}
puts "ONN Basys3 $stage completed; reports: $report_dir"
