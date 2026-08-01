# create_quartus_project.tcl - Create the baseline_cnn_smoke Quartus project.
#
# Equivalent to:
#     project_new -overwrite -family "Cyclone IV E" -part "EP4CE10F17C8" baseline_cnn_smoke
#
# plus:
#     TOP_LEVEL_ENTITY    = arithmetic_smoke_top
#     VERILOG_INPUT_VERSION = VERILOG_2001
#     VIRTUAL_PIN ON      on every top-level port (board pinout not frozen)
#
# The project files (.qpf / .qsf) are written to fpga/baseline_cnn/quartus/.
# Source paths in the .qsf are relative to the project directory so the
# project is portable.  Quartus databases / incremental caches / outputs are
# gitignored and never committed.

package require ::quartus::project

# Operate inside the quartus dir regardless of where quartus_sh was launched.
set script_dir   [file dirname [file normalize [info script]]]
set quartus_dir  [file join [file dirname $script_dir] "quartus"]
file mkdir $quartus_dir
cd $quartus_dir

set project_name "baseline_cnn_smoke"

project_new -overwrite \
    -family "Cyclone IV E" \
    -part   "EP4CE10F17C8" \
    $project_name

# TOP is the registered smoke wrapper (see baseline_cnn_smoke_top.v header:
# the Fitter rejects VIRTUAL_PIN on register-less ports, so one register
# stage is added around the pure-combinational arithmetic_smoke_top core).
set_global_assignment -name TOP_LEVEL_ENTITY baseline_cnn_smoke_top
set_global_assignment -name VERILOG_INPUT_VERSION VERILOG_2001

# RTL sources (relative paths -> portable .qsf).
set_global_assignment -name VERILOG_FILE ../rtl/requantize_u8.v
set_global_assignment -name VERILOG_FILE ../rtl/gap_div49.v
set_global_assignment -name VERILOG_FILE ../rtl/arithmetic_smoke_top.v
set_global_assignment -name VERILOG_FILE ../rtl/baseline_cnn_smoke_top.v

# baseline_cnn_smoke_top is NOT the final board-level top; every port
# (including the smoke-only virtual clock `clk`) is a virtual pin until the
# AC620 board pinout is frozen.  VIRTUAL_PIN is a *logic option* set with
# set_instance_assignment (set_location_assignment VIRTUAL_PIN would be
# treated by the Fitter as an illegal physical location).
foreach pin {clk acc multiplier shift gap_sum requant_q gap_q} {
    set_instance_assignment -name VIRTUAL_PIN ON -to $pin
}

export_assignments
project_close

puts "CREATE_PROJECT_OK project=$project_name family=Cyclone IV E part=EP4CE10F17C8"
