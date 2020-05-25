import pylidc as pl
import matplotlib.pyplot as plt

ann = pl.query(pl.Annotation).first()
i,j,k = ann.centroid
print (i)
print (j)
print (k)
print (int(k))
vol = ann.scan.to_volume()

plt.imshow(vol[:,:,int(k)], cmap=plt.cm.gray)
plt.plot(j, i, '.r', label="Nodule centroid")
plt.legend()
plt.show()
