# create_ac620_led_bringup_project.tcl - Create the AC620 LED bring-up Quartus
# project (the FIRST physical board project; no virtual pins).
#
# Equivalent to:
#     project_new -overwrite -family "Cyclone IV E" -part "EP4CE10F17C8" \
#         ac620_led_bringup
#
# plus:
#     TOP_LEVEL_ENTITY      = ac620_led_bringup_top
#     VERILOG_INPUT_VERSION = VERILOG_2001
#     SDC_FILE              = ac620_led_bringup.sdc  (50 MHz create_clock)
#
# Real AC620 V2 physical pin constraints (frozen 2026-08-02 from the board
# back-silkscreen; DO NOT guess these from other AC620 versions / web sources):
#     clk_50m -> PIN_E1   (50 MHz, 3.3-V LVTTL)
#     led[0]  -> PIN_A2   (3.3-V LVTTL)
#     led[1]  -> PIN_B3   (3.3-V LVTTL)
#     led[2]  -> PIN_A4   (3.3-V LVTTL)
#     led[3]  -> PIN_A3   (3.3-V LVTTL)
#
# This project is the physical board bring-up top, so VIRTUAL_PIN is NOT set on
# any port.  The .qpf / .qsf are written to fpga/baseline_cnn/quartus/ (relative
# paths in the .qsf keep the project portable); databases / caches / outputs are
# gitignored.

package require ::quartus::project

set script_dir   [file dirname [file normalize [info script]]]
set quartus_dir  [file join [file dirname $script_dir] "quartus"]
file mkdir $quartus_dir
cd $quartus_dir

set project_name "ac620_led_bringup"

project_new -overwrite \
    -family "Cyclone IV E" \
    -part   "EP4CE10F17C8" \
    $project_name

set_global_assignment -name TOP_LEVEL_ENTITY ac620_led_bringup_top
set_global_assignment -name VERILOG_INPUT_VERSION VERILOG_2001

# RTL source (single self-contained file; relative path -> portable .qsf).
set_global_assignment -name VERILOG_FILE ../rtl/ac620_led_bringup_top.v

# Real clock constraint (same directory as the .qsf).
set_global_assignment -name SDC_FILE ac620_led_bringup.sdc

# ---- real AC620 physical pins (frozen) ----
set_location_assignment PIN_E1 -to clk_50m
set_location_assignment PIN_A2 -to led[0]
set_location_assignment PIN_B3 -to led[1]
set_location_assignment PIN_A4 -to led[2]
set_location_assignment PIN_A3 -to led[3]

# ---- I/O standard: all ports 3.3-V LVTTL ----
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to clk_50m
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to led[0]
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to led[1]
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to led[2]
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to led[3]

export_assignments
project_close

puts "CREATE_AC620_LED_BRINGUP_PROJECT_OK project=$project_name family=Cyclone IV E part=EP4CE10F17C8"
