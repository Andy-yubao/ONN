# create_stem_conv_project.tcl - Create the stem_conv_smoke Quartus project.
#
# Equivalent to:
#     project_new -overwrite -family "Cyclone IV E" -part "EP4CE10F17C8" stem_conv_smoke
#
# plus:
#     TOP_LEVEL_ENTITY      = stem_conv_smoke_top  (registered smoke wrapper)
#     VERILOG_INPUT_VERSION = VERILOG_2001
#     VIRTUAL_PIN ON        on every top-level port (board pinout not frozen)
#
# The project files (.qpf / .qsf) are written to fpga/baseline_cnn/quartus/.
# Source paths in the .qsf are relative to the project directory so the
# project is portable.  Quartus databases / incremental caches / outputs are
# gitignored and never committed.
#
# This compiles the FULL serial stem engine (not the whole CNN): stem_conv_serial
# + sync_ram_u8 + sync_rom_s8 + sync_rom_s32 + requantize_u8.  Resource usage is
# therefore the stem single-MAC layer only.

package require ::quartus::project

set script_dir   [file dirname [file normalize [info script]]]
set quartus_dir  [file join [file dirname $script_dir] "quartus"]
file mkdir $quartus_dir
cd $quartus_dir

set project_name "stem_conv_smoke"

project_new -overwrite \
    -family "Cyclone IV E" \
    -part   "EP4CE10F17C8" \
    $project_name

set_global_assignment -name TOP_LEVEL_ENTITY stem_conv_smoke_top
set_global_assignment -name VERILOG_INPUT_VERSION VERILOG_2001

# RTL sources (relative paths -> portable .qsf).  stem_conv_serial pulls in
# requantize_u8 / sync_ram_u8 / sync_rom_s8 / sync_rom_s32 as instances; all
# six files are listed so a single compile pass sees the whole hierarchy.
set_global_assignment -name VERILOG_FILE ../rtl/requantize_u8.v
set_global_assignment -name VERILOG_FILE ../rtl/requantize_u8_pipe.v
set_global_assignment -name VERILOG_FILE ../rtl/sync_ram_u8.v
set_global_assignment -name VERILOG_FILE ../rtl/sync_rom_s8.v
set_global_assignment -name VERILOG_FILE ../rtl/sync_rom_s32.v
set_global_assignment -name VERILOG_FILE ../rtl/stem_conv_serial.v
set_global_assignment -name VERILOG_FILE ../rtl/stem_conv_smoke_top.v

# stem_conv_smoke_top is NOT the final board-level top; every port (including
# the smoke-only virtual clock `clk`) is a virtual pin until the AC620 board
# pinout is frozen.  VIRTUAL_PIN is a *logic option* set with
# set_instance_assignment (set_location_assignment VIRTUAL_PIN would be
# treated by the Fitter as an illegal physical location).
foreach pin {clk rst_n input_we input_waddr input_wdata start busy done \
             acc_valid q_valid dbg_qcnt dbg_qxor output_raddr output_rdata} {
    set_instance_assignment -name VIRTUAL_PIN ON -to $pin
}

export_assignments
project_close

puts "CREATE_STEM_CONV_PROJECT_OK project=$project_name family=Cyclone IV E part=EP4CE10F17C8"
