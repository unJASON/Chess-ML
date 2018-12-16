import ctypes
import shutil
import os
ll = ctypes.cdll.LoadLibrary
lib = ll("TFversionlib.so")
lib.startSelfPlay()
shutil.rmtree('chess')
os.mkdir('chess')