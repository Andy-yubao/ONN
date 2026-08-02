# ac620_led_bringup.sdc - Real board clock constraint for the AC620 bring-up top.
#
# The AC620 V2 board carries a 50 MHz crystal on PIN_E1 (3.3-V LVTTL), frozen
# from the board back-silkscreen on 2026-08-02.  Constraining the real clock
# keeps "Timing requirements not specified" out of the compile log and lets the
# timing analyzer report setup/hold slack and Fmax.
#
# This is a minimal board top (free-running LED counter, no reset / no PLL).
# A single primary clock is required; derive_clock_uncertainty fills in
# realistic on-chip clock uncertainty so the timing analyzer does not warn
# about missing uncertainty on the intra-clock transfers.

create_clock -name clk_50m -period 20.000 [get_ports {clk_50m}]
derive_clock_uncertainty
