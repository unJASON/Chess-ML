import ctypes
ll = ctypes.cdll.LoadLibrary
lib = ll("chess.so")
lib.foo(1, 3)