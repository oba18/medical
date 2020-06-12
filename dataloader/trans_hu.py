import pydicom
import os
import numpy as np
import glob
# Load the scans in given folder path

def load_scan(path):
    # slices = [pydicom.dcmread(path + "/" + s) for s in os.listdir(path)]
    slices = [pydicom.dcmread(path + s) for s in glob.glob(path + '*.dcm')]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    #image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)

patient = load_scan('/Users/masashi/Desktop/LIDC-IDRI-0001/01-01-2000-30178/3000566.000000-03192/')
hu = get_pixels_hu(patient)
print (hu.shape)
