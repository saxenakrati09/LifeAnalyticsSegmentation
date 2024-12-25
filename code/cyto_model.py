import numpy as np
import matplotlib.pyplot as plt
from cellpose import models, io
from cellpose.io import imread
import os

io.logger_setup()

# list of files
# PUT PATH TO YOUR FILES HERE!
files = os.listdir('../images/') #['../images/A-1.jpg']
imgs = [imread(f) for f in files]
nimg = len(imgs)