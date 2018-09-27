import os
import numpy as np

dir = '11KFingers/'

files = os.listdir(dir)
file_count = len(files)

all_samples = np.zeros((file_count, 28, 28, 3), dtype = 'uint8')
all_labels = np.zeros((file_count), dtype = 'uint8')

for i in range(file_count):
    filename = files[i]

    image = np.load(dir + filename)
    all_samples[i] = image

    if 'Finger' in filename:
        all_labels[i] = 1
    else:
        all_labels[i] = 0

np.save('X', all_samples)
np.save('Y', all_labels)
