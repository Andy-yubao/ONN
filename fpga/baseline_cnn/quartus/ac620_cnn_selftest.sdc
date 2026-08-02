# ac620_cnn_selftest.sdc - Real board clock constraint for the AC620 fixed-digit
# CNN self-test top.
#
# The AC620 V2 board carries a 50 MHz crystal on PIN_E1 (3.3-V LVTTL), frozen
# from the board back-silkscreen on 2026-08-02.  Constraining the real clock
# keeps "Timing requirements not specified" out of the compile log and lets the
# timing analyzer report setup/hold slack and Fmax.
#
# The self-test completes one ~1.53 M-cycle (30.6 ms) inference after POR, then
# free-runs its LED display, so a single primary clock is all that is required;
# derive_clock_uncertainty fills in realistic on-chip clock uncertainty for the
# intra-clock transfers.

create_clock -name clk_50m -period 20.000 [get_ports {clk_50m}]
derive_clock_uncertainty
