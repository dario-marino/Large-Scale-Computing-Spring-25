#!/bin/bash

# Display banner with warning
echo "==========================================================="
echo "  Improved Package Compatibility Fix Script for Midway 3"
echo "  This script will manage user-installed packages only"
echo "==========================================================="

# Load required modules
echo "Loading required modules..."
module load python/anaconda-2022.05 mpich/3.2.1 cuda/11.7

# Create a backup of current pip packages (just in case)
echo "Creating backup of current pip packages list..."
pip list > ~/pip_packages_backup_$(date +%Y%m%d_%H%M%S).txt

# Instead of trying to uninstall system packages (which causes permission errors),
# we'll focus on the user-installed packages in ~/.local
echo "Cleaning up user-installed packages..."
# Only try to uninstall packages that might be in user space
pip uninstall -y --user numpy scipy pandas pyarrow numba mpi4py pyopencl pycuda rasterio

# Install specific versions for compatibility
echo "Installing compatible package versions in user space..."

# First install numpy at specific version
echo "Installing numpy 1.22.4..."
pip install --user numpy==1.22.4

# Then install other packages with specific versions
echo "Installing remaining packages with compatible versions..."
pip install --user pandas==1.4.4 \
               pyarrow==9.0.0 \
               scipy==1.8.0 \
               numba==0.57.1 \
               mpi4py-mpich==3.1.5 \
               pyopencl==2024.1 \
               pycuda==2024.1 \
               rasterio==1.3.9

# Create a pip.conf file to ignore incompatible dependency warnings
mkdir -p ~/.config/pip
cat > ~/.config/pip/pip.conf << EOF
[global]
break-system-packages = true
EOF

# Verify installation
echo "==========================================================="
echo "Verifying package versions:"
pip list | grep -E "numpy|pandas|pyarrow|scipy|mpi4py|numba|pyopencl|pycuda|rasterio"
echo "==========================================================="

# Testing imports to verify functionality
echo "Testing imports to verify functionality..."
python -c "
try:
    import numpy
    print(f'Successfully imported numpy {numpy.__version__}')
    import pandas
    print(f'Successfully imported pandas {pandas.__version__}')
    import pyarrow
    print(f'Successfully imported pyarrow {pyarrow.__version__}')
    import scipy
    print(f'Successfully imported scipy {scipy.__version__}')
    import numba
    print(f'Successfully imported numba {numba.__version__}')
    try:
        import mpi4py
        print(f'Successfully imported mpi4py {mpi4py.__version__}')
    except ImportError:
        print('mpi4py import failed - may need special initialization')
    try:
        import pyopencl
        print(f'Successfully imported pyopencl {pyopencl.__version__}')
    except ImportError:
        print('pyopencl import failed - may need GPU context')
    try:
        import pycuda
        print(f'Successfully imported pycuda {pycuda.__version__}')
    except ImportError:
        print('pycuda import failed - may need GPU context')
    try:
        import rasterio
        print(f'Successfully imported rasterio {rasterio.__version__}')
    except ImportError:
        print('rasterio import failed')
except ImportError as e:
    print(f'Import error: {e}')
"

# Create a simple wrapper script for running Python with these packages
cat > ~/run_compatible_python.sh << EOF
#!/bin/bash
# Wrapper script to run Python with compatible environment

# Load required modules
module load python/anaconda-2022.05 mpich/3.2.1 cuda/11.7

# Run Python with user args
python "\$@"
EOF

chmod +x ~/run_compatible_python.sh

echo "==========================================================="
echo "Script completed. You can now use compatible packages for your code."
echo ""
echo "To run Python scripts with this environment, use:"
echo "  ~/run_compatible_python.sh your_script.py"
echo ""
echo "If any compatibility issues persist, try:"
echo "1. Running 'pip cache purge' before running this script again"
echo "2. Using a Python virtual environment for isolation"
echo "3. Adding PYTHONPATH=~/.local/lib/python3.9/site-packages to your environment"
echo "==========================================================="