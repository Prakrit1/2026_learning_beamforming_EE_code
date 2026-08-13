# 3GPP TR 38.821 parameters -- LEO-600 scenario

Source: 3GPP TR 38.821 v16.2.0, Set-1 (LEO-600, S-band, 2 GHz, downlink,
handheld UE),

| Parameter | Value | `config.py` field |
|---|---|---|
| Orbit altitude | 600 km | `altitude_orbit` |
| Frequency | 2 GHz (S-band) | `freq` |
| Satellite Tx gain | 30 dBi | `sat_gain_dBi` |
| Bandwidth | 30 MHz | (in `noise_power_watt` formula) |
| EIRP density | 34 dBW/MHz | -- |
| Radiated power (derived: EIRP density + 10log10(BW) - gain) | ~75 W | `power_constraint_watt` |
| 3dB beamwidth | 4.4127 deg | -- (our array: ~2.7-3.6 deg, same order) |
| Equivalent aperture | 2 m | -- (our ULA: ~3.4 m, 16 el x 3lambda/2) |
| UE noise figure | 7 dB | (in `noise_power_watt` formula) |
| UE antenna/ambient temp | 290 K | (in `noise_power_watt` formula) |
| UE Tx gain | 0 dBi | `user_gain_dBi` |
| Target elevation (Table 6.1.3.2-1, p. 52) | 30 deg | `EE_TARGET_ELEVATION_DEG` (optional; default is nadir) |

SSSS
