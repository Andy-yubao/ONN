# create_conv23_project.tcl - Create the conv23_smoke Quartus project.
#
# Equivalent to:
#     project_new -overwrite -family "Cyclone IV E" -part "EP4CE10F17C8" conv23_smoke
#
# plus:
#     TOP_LEVEL_ENTITY      = conv23_smoke_top  (registered smoke wrapper)
#     VERILOG_INPUT_VERSION = VERILOG_2001
#     VIRTUAL_PIN ON        on every top-level port (board pinout not frozen)
#
# The project files (.qpf / .qsf) are written to fpga/baseline_cnn/quartus/.
# Source paths in the .qsf are relative to the project directory so the
# project is portable.  Quartus databases / incremental caches / outputs are
# gitignored and never committed.
#
# This compiles the FULL shared conv2/conv3 engine:
#   conv_u8_serial (both conv2/conv3 weight ROMs + both bias ROMs +
#   requantize_u8) + sync_ram_u8 / sync_rom_s8 / sync_rom_s32 templates +
#   an external feature-map RAM (3136 x 8) in conv23_smoke_top.
# The engine is verified standalone; conv2+pool2 integration is covered by the
# Questa golden testbenches (tb_conv2_pool2 / tb_conv3_serial).

package require ::quartus::project

set script_dir   [file dirname [file normalize [info script]]]
set quartus_dir  [file join [file dirname $script_dir] "quartus"]
file mkdir $quartus_dir
cd $quartus_dir

set project_name "conv23_smoke"

project_new -overwrite \
    -family "Cyclone IV E" \
    -part   "EP4CE10F17C8" \
    $project_name

set_global_assignment -name TOP_LEVEL_ENTITY conv23_smoke_top
set_global_assignment -name VERILOG_INPUT_VERSION VERILOG_2001

# RTL sources (relative paths -> portable .qsf).  conv23_smoke_top pulls in
# conv_u8_serial (which instantiates requantize_u8 + the four ROMs) and the
# sync_ram_u8 feature-map RAM; all six files are listed so a single compile
# pass sees the whole hierarchy.
set_global_assignment -name VERILOG_FILE ../rtl/requantize_u8.v
set_global_assignment -name VERILOG_FILE ../rtl/sync_ram_u8.v
set_global_assignment -name VERILOG_FILE ../rtl/sync_rom_s8.v
set_global_assignment -name VERILOG_FILE ../rtl/sync_rom_s32.v
set_global_assignment -name VERILOG_FILE ../rtl/conv_u8_serial.v
set_global_assignment -name VERILOG_FILE ../rtl/conv23_smoke_top.v

# conv23_smoke_top is NOT the final board-level top; every port (including the
# smoke-only virtual clock `clk`) is a virtual pin until the AC620 board
# pinout is frozen.  VIRTUAL_PIN is a *logic option* set with
# set_instance_assignment (set_location_assignment VIRTUAL_PIN would be
# treated by the Fitter as an illegal physical location).  Bus names apply to
# every bit of the bus.
foreach pin {clk rst_n start layer_sel fm_we fm_waddr fm_wdata \
             busy done acc_valid q_valid dbg_qcnt dbg_qxor} {
    set_instance_assignment -name VIRTUAL_PIN ON -to $pin
}

export_assignments
project_close

puts "CREATE_CONV23_PROJECT_OK project=$project_name family=Cyclone IV E part=EP4CE10F17C8"
