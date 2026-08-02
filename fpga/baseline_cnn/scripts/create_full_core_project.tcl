# create_full_core_project.tcl - Create the baseline_cnn_core_smoke Quartus project.
#
# Equivalent to:
#     project_new -overwrite -family "Cyclone IV E" -part "EP4CE10F17C8" baseline_cnn_core_smoke
#
# plus:
#     TOP_LEVEL_ENTITY      = baseline_cnn_core_smoke_top  (registered smoke wrapper)
#     VERILOG_INPUT_VERSION = VERILOG_2001
#     VIRTUAL_PIN ON        on every top-level port (board pinout not frozen)
#
# The project files (.qpf / .qsf) are written to fpga/baseline_cnn/quartus/.
# Source paths in the .qsf are relative to the project directory so the project
# is portable.  Quartus databases / incremental caches / outputs are gitignored
# and never committed.
#
# This compiles the COMPLETE pure-compute BaselineCNN core:
#   stem_pool1_pipeline (stem_conv_serial + stream maxpool + pool1 RAM),
#   conv_u8_serial (shared conv2/conv3 engine, both weight/bias ROMs),
#   the parameterized pool2 maxpool + pool2 RAM,
#   gap_stream_u8 (reusing gap_div49),
#   fc_argmax_serial (fc weight/bias ROMs),
#   baseline_cnn_core (the controller) + the compressed smoke wrapper.
# Feature-map storage is limited to input_q / pool1_q / pool2_q RAM + gap_mem
# registers: NO complete stem_q / conv2_q / conv3_q RAM is instantiated.

package require ::quartus::project

set script_dir   [file dirname [file normalize [info script]]]
set quartus_dir  [file join [file dirname $script_dir] "quartus"]
file mkdir $quartus_dir
cd $quartus_dir

set project_name "baseline_cnn_core_smoke"

project_new -overwrite \
    -family "Cyclone IV E" \
    -part   "EP4CE10F17C8" \
    $project_name

set_global_assignment -name TOP_LEVEL_ENTITY baseline_cnn_core_smoke_top
set_global_assignment -name VERILOG_INPUT_VERSION VERILOG_2001

# RTL sources (relative paths -> portable .qsf).  baseline_cnn_core_smoke_top
# pulls in baseline_cnn_core, which instantiates every sub-block; all files are
# listed so a single compile pass sees the whole hierarchy.
set_global_assignment -name VERILOG_FILE ../rtl/requantize_u8.v
set_global_assignment -name VERILOG_FILE ../rtl/requantize_u8_pipe.v
set_global_assignment -name VERILOG_FILE ../rtl/sync_ram_u8.v
set_global_assignment -name VERILOG_FILE ../rtl/sync_rom_s8.v
set_global_assignment -name VERILOG_FILE ../rtl/sync_rom_s32.v
set_global_assignment -name VERILOG_FILE ../rtl/stem_conv_serial.v
set_global_assignment -name VERILOG_FILE ../rtl/maxpool2x2_stream.v
set_global_assignment -name VERILOG_FILE ../rtl/stem_pool1_pipeline.v
set_global_assignment -name VERILOG_FILE ../rtl/conv_u8_serial.v
set_global_assignment -name VERILOG_FILE ../rtl/gap_div49.v
set_global_assignment -name VERILOG_FILE ../rtl/gap_stream_u8.v
set_global_assignment -name VERILOG_FILE ../rtl/fc_argmax_serial.v
set_global_assignment -name VERILOG_FILE ../rtl/baseline_cnn_core.v
set_global_assignment -name VERILOG_FILE ../rtl/baseline_cnn_core_smoke_top.v

# baseline_cnn_core_smoke_top is NOT the final board-level top; every port
# (including the smoke-only virtual clock `clk`) is a virtual pin until the
# AC620 board pinout is frozen.  VIRTUAL_PIN is a *logic option* set with
# set_instance_assignment (set_location_assignment VIRTUAL_PIN would be treated
# by the Fitter as an illegal physical location).  Bus names apply to every bit
# of the bus.
foreach pin {clk rst_n start input_we input_waddr input_wdata \
             busy done prediction dbg_stage dbg_cnt dbg_xor \
             dbg_last_addr dbg_last_val} {
    set_instance_assignment -name VIRTUAL_PIN ON -to $pin
}

export_assignments
project_close

puts "CREATE_FULL_CORE_PROJECT_OK project=$project_name family=Cyclone IV E part=EP4CE10F17C8"
