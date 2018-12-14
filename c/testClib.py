import ctypes
ll = ctypes.cdll.LoadLibrary
lib = ll("TFversionlib.so")
lib.startSelfPlay()