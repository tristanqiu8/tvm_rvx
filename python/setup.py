# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, exec-used
"""Setup TVM package."""
import os
import pathlib
import shutil
import sys

from setuptools import find_packages
from setuptools.dist import Distribution

# need to use distutils.core for correct placement of cython dll
if "--inplace" in sys.argv:
    from distutils.core import setup
    from distutils.extension import Extension
else:
    from setuptools import setup
    from setuptools.extension import Extension

CURRENT_DIR = os.path.dirname(__file__)
FFI_MODE = os.environ.get("TVM_FFI", "auto")
CONDA_BUILD = os.getenv("CONDA_BUILD") is not None
INPLACE_BUILD = "--inplace" in sys.argv


def get_lib_path():
    """Get library path, name and version"""
    # We can not import `libinfo.py` in setup.py directly since __init__.py
    # Will be invoked which introduces dependencies
    libinfo_py = os.path.join(CURRENT_DIR, "./tvm/libinfo.py")
    libinfo = {"__file__": libinfo_py}
    exec(compile(open(libinfo_py, "rb").read(), libinfo_py, "exec"), libinfo, libinfo)
    version = libinfo["__version__"]
    if not CONDA_BUILD and not INPLACE_BUILD:
        lib_path = libinfo["find_lib_path"]()
        libs = [lib_path[0]]
        if "runtime" not in libs[0]:
            for name in lib_path[1:]:
                if "runtime" in name:
                    libs.append(name)
                    break

        # Add byoc shared libraries, if present
        for name in lib_path:
            if "3rdparty" in name:
                libs.append(name)

        # Add tvmc configuration json files
        for name in lib_path:
            candidate_path = os.path.abspath(os.path.join(os.path.dirname(name), "..", "configs"))
            if os.path.isdir(candidate_path):
                libs.append(candidate_path)
                break

        for dir in [
            "3rdparty",
            "jvm",
            "web",
            "rust",
            "golang",
            "include",
            "src",
            "cmake",
            "CMakeLists.txt",
        ]:
            for name in lib_path:
                candidate_path = os.path.abspath(os.path.join(os.path.dirname(name), "..", dir))
                if os.path.exists(candidate_path):
                    libs.append(candidate_path)
                    if dir == "3rdparty":
                        # remove large files
                        _remove_path(os.path.join(candidate_path, "cutlass", "docs"))
                        _remove_path(os.path.join(candidate_path, "cutlass", "media"))
                        _remove_path(
                            os.path.join(candidate_path, "cutlass_fpA_intB_gemm", "cutlass", "docs")
                        )
                        _remove_path(
                            os.path.join(
                                candidate_path, "cutlass_fpA_intB_gemm", "cutlass", "media"
                            )
                        )
                        _remove_path(
                            os.path.join(candidate_path, "libflash_attn", "cutlass", "docs")
                        )
                        _remove_path(
                            os.path.join(candidate_path, "libflash_attn", "cutlass", "media")
                        )
                    break
    else:
        libs = None

    return libs, version


def git_describe_version(original_version):
    """Get git describe version."""
    ver_py = os.path.join(CURRENT_DIR, "..", "version.py")
    libver = {"__file__": ver_py}
    exec(compile(open(ver_py, "rb").read(), ver_py, "exec"), libver, libver)
    _, gd_version = libver["git_describe_version"]()
    if gd_version != original_version and "--inplace" not in sys.argv:
        print("Use git describe based version %s" % gd_version)
    return gd_version


def _remove_path(path):
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)


LIB_LIST, __version__ = get_lib_path()
__version__ = git_describe_version(__version__)

if not CONDA_BUILD and not INPLACE_BUILD:
    # Wheel cleanup
    for path in LIB_LIST:
        libname = os.path.basename(path)
        _remove_path(f"tvm/{libname}")


def config_cython():
    """Try to configure cython and return cython configuration"""
    # Enforce cython unless FFI_MODE is explicitly set to ctypes
    # we might consider fully converge to cython later
    if FFI_MODE == "ctypes":
        return []
    try:
        from Cython.Build import cythonize

        # for python 3.12+, use limited API for future compact
        limited_api_kwargs = {}
        if sys.version_info >= (3, 12):
            limited_api_kwargs = {
                "define_macros": [
                    ("Py_LIMITED_API", 0x030C0000),
                ],
                "py_limited_api": True,
            }

        ret = []
        extra_compile_args = ["-std=c++17", "-DDMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>"]
        if os.name == "nt":
            library_dirs = ["tvm", "../build/Release", "../build"]
            libraries = ["tvm"]
            extra_compile_args = [
                "/std:c++17",
                "/D DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>",
            ]
            # library is available via conda env.
            if CONDA_BUILD:
                library_dirs = [os.environ["LIBRARY_LIB"]]
        else:
            library_dirs = None
            libraries = None

        # the latest ffi source
        for fn in os.listdir("tvm/ffi/cython"):
            if not fn.endswith(".pyx"):
                continue
            ret.append(
                Extension(
                    f"tvm.ffi.{fn[:-4]}",
                    ["tvm/ffi/cython/%s" % fn],
                    include_dirs=[
                        "../ffi/include/",
                        "../ffi/3rdparty/dlpack/include",
                    ],
                    extra_compile_args=extra_compile_args,
                    library_dirs=library_dirs,
                    libraries=libraries,
                    language="c++",
                    **limited_api_kwargs,
                )
            )
        return cythonize(ret, compiler_directives={"language_level": 3})
    except ImportError as error:
        raise RuntimeError("Cython is not installed, please pip install cython")


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

    def is_pure(self):
        return False


setup_kwargs = {}
if not CONDA_BUILD and not INPLACE_BUILD:
    with open("MANIFEST.in", "w") as fo:
        for path in LIB_LIST:
            if os.path.isfile(path):
                shutil.copy(path, os.path.join(CURRENT_DIR, "tvm"))
                _, libname = os.path.split(path)
                fo.write(f"include tvm/{libname}\n")

            if os.path.isdir(path):
                _, libname = os.path.split(path)
                shutil.copytree(path, os.path.join(CURRENT_DIR, "tvm", libname))
                fo.write(f"recursive-include tvm/{libname} *\n")

    setup_kwargs = {"include_package_data": True}


def long_description_contents():
    with open(pathlib.Path(CURRENT_DIR).resolve().parent / "README.md", encoding="utf-8") as readme:
        description = readme.read()

    return description


# Temporarily add this directory to the path so we can import the requirements generator
# tool.
sys.path.insert(0, os.path.dirname(__file__))
import gen_requirements

sys.path.pop(0)

requirements = gen_requirements.join_requirements()
extras_require = {
    piece: deps for piece, (_, deps) in requirements.items() if piece not in ("all", "core")
}

setup(
    name="tvm",
    version=__version__,
    description="TVM: An End to End Tensor IR/DSL Stack for Deep Learning Systems",
    long_description=long_description_contents(),
    long_description_content_type="text/markdown",
    url="https://tvm.apache.org/",
    download_url="https://github.com/apache/tvm/tags",
    author="Apache TVM",
    license="Apache",
    # See https://pypi.org/classifiers/
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
    ],
    keywords="machine learning",
    zip_safe=False,
    install_requires=requirements["core"][1],
    extras_require=extras_require,
    packages=find_packages(),
    package_dir={"tvm": "tvm"},
    distclass=BinaryDistribution,
    ext_modules=config_cython(),
    **setup_kwargs,
)


if not CONDA_BUILD and not INPLACE_BUILD:
    # Wheel cleanup
    os.remove("MANIFEST.in")
    for path in LIB_LIST:
        libname = os.path.basename(path)
        _remove_path(f"tvm/{libname}")
