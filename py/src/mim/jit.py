import platform as pf
import subprocess
import os
from ctypes import CDLL, cdll
from abc import ABC

class JIT(ABC):

    def __init__(self, so_name: str):
        self.so_name = so_name
        self.ll_name = so_name + ".ll"
        self._lib = None

    def _get_so_path(self) -> str:
        ext = ".dll" if pf.system() == "Windows" else ".so"
        return os.path.join(".", self.so_name + ext)

    def _compile_so(self):
        so_path = self._get_so_path()
        subprocess.run(
            ["clang", self.ll_name, "-o", so_path, "-shared", "-Wno-override-module"],
            check=True,
        )
        return so_path

    def compile_and_load(self) -> CDLL:
        so_path = self._compile_so()
        self._lib = cdll.LoadLibrary(so_path)
        return self._lib
