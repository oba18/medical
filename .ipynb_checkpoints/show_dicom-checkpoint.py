import matplotlib.pyplot as plt
import pydicom

ds = pydicom.dcmread('/Volumes/masashi/workspace/0_KML/0_medical/0_data/LIDC-IDRI/LIDC-IDRI-0078/01-01-2000-01207/3178.000000-W Chest PA-63997/1-1.dcm')
plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
