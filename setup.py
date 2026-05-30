from setuptools import setup
from torch.utils import cpp_extension

# Python 3.12 stable limited API.
PY_LIMITED_API = "0x030c0000"

# Torch 2.11 stable API.
# Encoding is 0x<major:2><minor:2><patch:2>00000000.
TORCH_TARGET_VERSION = "0x020b000000000000"

abi_defines = [
    f"-DPy_LIMITED_API={PY_LIMITED_API}",
    f"-DTORCH_TARGET_VERSION={TORCH_TARGET_VERSION}",
]

setup(
    name="asr",
    ext_modules=[
        cpp_extension.CUDAExtension(
            name="asr.loss._C",
            sources=[
                "src/asr/loss/csrc/ctc_bindings.cpp",
                "src/asr/loss/csrc/ctc.cu",
                "src/asr/loss/csrc/rnnt_bindings.cpp",
                "src/asr/loss/csrc/rnnt.cu",
                "src/asr/loss/csrc/tdt_bindings.cpp",
                "src/asr/loss/csrc/tdt.cu",
            ],
            extra_compile_args={
                "cxx": abi_defines,
                "nvcc": [*abi_defines, "-DUSE_CUDA", "-O3", "--use_fast_math"],
            },
            extra_link_args=["-lcublas"],
            py_limited_api=True,
        ),
        cpp_extension.CppExtension(
            name="asr.decode._C",
            sources=[
                "src/asr/decode/csrc/ctc.cpp",
            ],
            extra_compile_args={"cxx": [*abi_defines, "-O3", "-fopenmp"]},
            extra_link_args=["-fopenmp"],
            py_limited_api=True,
        ),
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    # One wheel for all CPython >= 3.12.
    options={"bdist_wheel": {"py_limited_api": "cp312"}},
)
