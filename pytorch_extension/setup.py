"""
PyTorch extension setup script for custom CUDA matrix multiplication kernels.
This integrates our CUDA implementations with PyTorch's autograd system.
"""

from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

# Check CUDA availability
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This extension requires CUDA support.")

# Define the extension
ext_modules = [
    CUDAExtension(
        name='matmul_cuda',
        sources=[
            'matmul_cuda.cpp',
            'matmul_cuda_kernels.cu'
        ],
        include_dirs=[
            # Add any additional include directories here
        ],
        extra_compile_args={
            'cxx': ['-O3'],
            'nvcc': [
                '-O3',
                '--use_fast_math',
                '-gencode=arch=compute_70,code=sm_70',  # V100
                '-gencode=arch=compute_75,code=sm_75',  # RTX 20xx
                '-gencode=arch=compute_80,code=sm_80',  # A100
                '-gencode=arch=compute_86,code=sm_86',  # RTX 30xx
                '-gencode=arch=compute_89,code=sm_89',  # RTX 40xx
            ]
        }
    )
]

setup(
    name='matmul_cuda',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
    author='Fast-Matmul',
    description='Custom CUDA matrix multiplication kernels for PyTorch',
    long_description='High-performance matrix multiplication implementations optimized for ML inference workloads',
    python_requires='>=3.7',
    install_requires=[
        'torch>=1.8.0',
        'numpy>=1.19.0'
    ]
)
