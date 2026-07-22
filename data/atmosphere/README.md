# Atmospheric input data

Provider directories separate immutable source material (`raw`) from
TPeanuts canonical tables (`flux`, `production`) and provenance (`metadata`).

The canonical flux schema is long-form:

`energy_GeV, cos_zenith, [azimuth_deg], [altitude_km], particle, flux`

Flux units are mandatory metadata because provider conventions differ. Honda
is the configured default. Bartol remains unavailable until an original
BGLRS table is supplied; Honda values are never used as a silent substitute.
