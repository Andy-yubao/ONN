# create_maxpool_project.tcl - Create the maxpool_smoke Quartus project.
#
# Equivalent to:
#     project_new -overwrite -family "Cyclone IV E" -part "EP4CE10F17C8" maxpool_smoke
#
# plus:
#     TOP_LEVEL_ENTITY      = maxpool_smoke_top  (registered smoke wrapper)
#     VERILOG_INPUT_VERSION = VERILOG_2001
#     VIRTUAL_PIN ON        on every top-level port (board pinout not frozen)
#
# The project files (.qpf / .qsf) are written to fpga/baseline_cnn/quartus/.
# Source paths in the .qsf are relative to the project directory so the
# project is portable.  Quartus databases / incremental caches / outputs are
# gitignored and never committed.
#
# This compiles ONLY the streaming 2x2 max-pool primitive
# (maxpool2x2_stream + maxpool_smoke_top).  It is pure logic: row_buffer is a
# 28x8 register array with combinational reads (too small and read asynchronously
# to map to an M9K), no multipliers, no dividers, no RAM/ROM, no vendor IP.

package require ::quartus::project

set script_dir   [file dirname [file normalize [info script]]]
set quartus_dir  [file join [file dirname $script_dir] "quartus"]
file mkdir $quartus_dir
cd $quartus_dir

set project_name "maxpool_smoke"

project_new -overwrite \
    -family "Cyclone IV E" \
    -part   "EP4CE10F17C8" \
    $project_name

set_global_assignment -name TOP_LEVEL_ENTITY maxpool_smoke_top
set_global_assignment -name VERILOG_INPUT_VERSION VERILOG_2001

# RTL sources (relative paths -> portable .qsf).  Only the primitive and its
# smoke wrapper: no sync_ram/sync_rom templates are needed.
set_global_assignment -name VERILOG_FILE ../rtl/maxpool2x2_stream.v
set_global_assignment -name VERILOG_FILE ../rtl/maxpool_smoke_top.v

# maxpool_smoke_top is NOT the final board-level top; every port (including
# the smoke-only virtual clock `clk`) is a virtual pin until the AC620 board
# pinout is frozen.  VIRTUAL_PIN is a *logic option* set with
# set_instance_assignment (set_location_assignment VIRTUAL_PIN would be
# treated by the Fitter as an illegal physical location).  Bus names apply to
# every bit of the bus.
foreach pin {clk rst_n start in_valid in_q busy out_valid out_addr out_q done \
             dbg_incnt dbg_outcnt} {
    set_instance_assignment -name VIRTUAL_PIN ON -to $pin
}

export_assignments
project_close

puts "CREATE_MAXPOOL_PROJECT_OK project=$project_name family=Cyclone IV E part=EP4CE10F17C8"
