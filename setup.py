import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

def get_gpu_arch_flags():
    try:
        major = torch.cuda.get_device_capability()[0]
        return [f'-gencode=arch=compute_{major}0,code=sm_{major}0']
    except Exception as e:
        print(f"Error while detecting GPU architecture: {e}")
        return []

arch_flags = get_gpu_arch_flags()

setup(
    name="rn3ext",
    version="0.0.1",
    packages=find_packages(),
    description="Rnn for the next century",
    author="Doraemonzzz",
    author_email="doraemon_zzz@163.com",
    url="https://github.com/Doraemonzzz/rnnext",
    ext_modules=[
        CUDAExtension(
            "rn3ext.wkv_cuda",
            sources=["rnn/rwkv/cuda/wkv_cuda.cu", "rnn/rwkv/cuda/wkv_op.cpp",],
            extra_compile_args={
                'cxx': ['-O2', '-std=c++14', '-D_GLIBCXX_USE_CXX11_ABI=0'],
                'nvcc': ['-O2', '-std=c++14', '-D_GLIBCXX_USE_CXX11_ABI=0'] + arch_flags
            }
        ),
        CUDAExtension(
            "rn3ext.lru_cuda",
            sources=["rnn/lru/cuda/lru_cuda_kernel.cu", "rnn/lru/cuda/lru_cuda.cpp",],
            extra_compile_args={
                'cxx': ['-O2', '-std=c++14', '-D_GLIBCXX_USE_CXX11_ABI=0'],
                'nvcc': ['-O2', '-std=c++14', '-D_GLIBCXX_USE_CXX11_ABI=0'] + arch_flags
            }
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False),},
    install_requires=["torch", "einops",],
    keywords=["artificial intelligence", "sequential model",],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
    ],
)
