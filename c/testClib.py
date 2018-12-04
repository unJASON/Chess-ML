import ctypes
ll = ctypes.cdll.LoadLibrary
lib = ll("LocalPlay.so")
lib.LocalSimulate()