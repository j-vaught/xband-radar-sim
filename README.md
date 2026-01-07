# xband-radar-sim

**X-Band Pulse Compression Radar Electromagnetic Simulation Framework**

A Python framework for end-to-end electromagnetic simulation of marine radar systems, specifically modeling the Furuno DRS4D-NXT solid-state X-band radar.

## Features

- **Antenna Simulation** (openEMS): Slotted waveguide array pattern generation
- **Wave Propagation**: CPU ray tracing with Numba JIT acceleration
- **Surface Scattering**: Physical Optics RCS, analytical formulas (Mie, corner reflector)
- **Signal Processing**: LFM chirp waveforms, matched filtering, CFAR detection

## Specifications

| Parameter | Value |
|-----------|-------|
| Frequency | 9.41 GHz (X-band) |
| Wavelength | 31.9 mm |
| TX Power | 25 W (solid-state) |
| Horizontal Beamwidth | 3.9° |
| Vertical Beamwidth | 25° |
| Target | 3" corner reflector (RCS: 0.082 m²) |

## Quick Start

```bash
# Clone and setup
git clone https://github.com/j-vaught/xband-radar-sim.git
cd xband-radar-sim
python3 -m venv venv
source venv/bin/activate
pip install -e .

# Activate environment (includes openEMS paths)
source setup_env.sh

# Run simulation
python -c "
from pipeline import run_simple_simulation
result = run_simple_simulation(target_range_m=2000, target_rcs_m2=0.082)
print(f'SNR: {result.snr_db:.1f} dB, Detections: {len(result.detected_targets)}')
"
```

## Project Structure

```
xband-radar-sim/
├── src/
│   ├── config.py           # RadarConfig dataclass
│   ├── interfaces.py       # Data contracts (AntennaPattern, RayBundle, etc.)
│   ├── pipeline.py         # End-to-end simulation
│   ├── antenna/            # openEMS waveguide/slot array
│   ├── propagation/        # Ray tracing, path loss
│   ├── scattering/         # RCS computation (PO, analytical)
│   └── signal/             # Waveforms, matched filter, CFAR
├── tests/                  # pytest test suite
├── config/                 # YAML configurations
└── docs/                   # Documentation
```

## Requirements

- Python 3.10+
- NumPy, SciPy, Matplotlib
- openEMS (FDTD solver) - see installation below
- Numba (optional, for JIT acceleration)

### openEMS Installation

openEMS must be built from source on macOS:

```bash
# Dependencies (Homebrew)
brew install cmake boost hdf5 vtk cgal

# Build openEMS (see setup_env.sh for full paths)
source setup_env.sh
```

## License

MIT License

## References

- Furuno DRS4D-NXT specifications
- openEMS FDTD solver: https://openems.de
