import src.datafile as datafile
import matplotlib.pyplot as plt
import src.classifier as classifier
import numpy as np
import rich

path = "data/example.hdf5"
df = datafile.DataFile(path)
amplitudes, times, attrs = df.read_timeseries(83)
#df.read_classification(classifier.ClassifyMethod.fixed_library, classifier.BasisMethod.admd)
fig, ax = plt.subplots()
N_site, N_time = amplitudes.shape
for row in amplitudes:
  ax.plot(times, np.real(row))

plt.show()
rich.print(times.shape)