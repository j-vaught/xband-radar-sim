#!/bin/bash
# openEMS Environment Setup Script
# Source this file to activate the radar simulation environment

# Activate virtualenv
if [ -f "$(dirname "${BASH_SOURCE[0]}")/venv/bin/activate" ]; then
    source "$(dirname "${BASH_SOURCE[0]}")/venv/bin/activate"
    echo "✓ Virtualenv activated"
else
    echo "⚠ Virtualenv not found, run: python3 -m venv venv"
fi

# openEMS libraries
export DYLD_LIBRARY_PATH="$HOME/.local/lib:/opt/homebrew/lib:$DYLD_LIBRARY_PATH"
export PATH="$HOME/.local/bin:/opt/homebrew/bin:$PATH"

# Python module paths
export PYTHONPATH="$HOME/openEMS-Project/openEMS/python:$HOME/openEMS-Project/CSXCAD/python:$PYTHONPATH"

# openEMS installation path (for build scripts)
export OPENEMS_INSTALL_PATH="$HOME/.local"

# Verify imports
python3 -c "
import CSXCAD
import openEMS
from openEMS.physical_constants import C0
print(f'✓ openEMS ready (C0 = {C0:.0f} m/s)')
" 2>/dev/null && echo "✓ All imports verified" || echo "⚠ Some imports failed"

echo ""
echo "Environment ready! Available commands:"
echo "  openEMS --help    # FDTD solver"
echo "  nf2ff --help      # Near-to-far-field"
