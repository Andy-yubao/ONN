# check_device.tcl - Verify the frozen target device is recognised by this
# Quartus Prime installation.
#
# Creates a throwaway project under fpga/baseline_cnn/sim/work/device_check
# (gitignored), asks Quartus to create it for the frozen family/part, then
# reads back the canonical family/part strings from the device database.
# If the part does not exist, project_new fails and quartus_sh exits non-zero.
#
# Prints:
#   CHECK_DEVICE family=<...> part=<...>
#   CHECK_DEVICE_OK | CHECK_DEVICE_FAIL

package require ::quartus::project

set family "Cyclone IV E"
set part   "EP4CE10F17C8"

# throwaway project location (computed from this script's own path).
set script_dir [file dirname [file normalize [info script]]]
set check_dir  [file join [file dirname $script_dir] "sim" "work" "device_check"]
file delete -force $check_dir
file mkdir $check_dir
cd $check_dir

project_new -overwrite -family $family -part $part device_check

set got_family [get_global_assignment -name FAMILY]
set got_device [get_global_assignment -name DEVICE]

puts "CHECK_DEVICE family=$got_family part=$got_device"
if {[string equal -nocase $got_family $family] && [string equal -nocase $got_device $part]} {
    puts "CHECK_DEVICE_OK"
} else {
    puts "CHECK_DEVICE_FAIL: expected family=$family part=$part"
}

project_close -dont_export_assignments
file delete -force $check_dir
