# Program the existing Basys3 bitstream and verify configuration status.
set root [file normalize [file join [file dirname [info script]] .. .. ..]]
set bitfile [file join $root fpga basys3 reports onn_basys3.bit]
if {![file exists $bitfile]} {
    error "Basys3 bitstream missing: $bitfile"
}

open_hw_manager
connect_hw_server -url localhost:3121
set targets [get_hw_targets]
if {[llength $targets] != 1} {
    error "Expected one JTAG target, found [llength $targets]: $targets"
}
open_hw_target [lindex $targets 0]
set devices [get_hw_devices]
if {[llength $devices] != 1 || ![string match xc7a35t* [lindex $devices 0]]} {
    error "Expected one xc7a35t device, found: $devices"
}

set device [lindex $devices 0]
set_property PROGRAM.FILE $bitfile $device
program_hw_devices $device
refresh_hw_device $device

foreach {property expected} {
    REGISTER.CONFIG_STATUS.BIT14_DONE_PIN 1
    REGISTER.CONFIG_STATUS.BIT00_CRC_ERROR 0
    REGISTER.CONFIG_STATUS.BIT15_IDCODE_ERROR 0
} {
    set actual [get_property $property $device]
    puts "$property=$actual"
    if {$actual ne $expected} {
        error "Configuration check failed: $property expected $expected, got $actual"
    }
}
puts "PASS Basys3 programmed: $device"
close_hw_target
disconnect_hw_server
close_hw_manager
