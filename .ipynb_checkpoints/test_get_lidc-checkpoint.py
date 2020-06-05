import numpy as np
import pylidc as pl

def get_scan():
    scan = pl.query(pl.Scan).filter(pl.Annotation.texture == 1)
    return scan.all()

def get_texture_1():
    texture_1 = pl.query(pl.Annotation).filter(pl.Annotation.texture == 1)
    return texture_1.all()

def get_scan2vol(scan):
    return scan.to_volume()

print (get_scan())
print ('==================')
print (get_texture_1())
# print ('==================')
# print (get_scan2vol(get_scan()))

