import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid warnings in headless environments
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from LucasKanadeAffine import LucasKanadeAffine
from file_utils import mkdir_if_missing

# NOTE: Either change data_name through command line or directly in code
import sys
data_name = sys.argv[1] if len(sys.argv) > 1 else 'car2'
# data_name = 'landing'      # could choose from (car1, car2, landing)

do_display = int(sys.argv[2]) if len(sys.argv) > 2 else 1

# load data name
data = np.load('../data/%s.npy' % data_name)

# obtain the initial rect with format (x1, y1, x2, y2)
if data_name == 'car1':
    initial = np.array([170, 130, 290, 250])
elif data_name == 'car2':
    initial = np.array([59,116,145,151])
elif data_name == 'landing':
    initial = np.array([440, 80, 560, 140])
else:
    assert False, 'the data name must be one of (car1, car2, landing)'

numFrames = data.shape[2]
w = initial[2] - initial[0]
h = initial[3] - initial[1]

# Load over frames
rects = []
rects.append(initial)
fig = plt.figure(1)
ax = fig.add_subplot(111)
for i in range(numFrames-1):
    print("frame****************", i)
    It = data[:,:,i]
    It1 = data[:,:,i+1]
    rect = rects[i]

    # Run algorithm
    M = LucasKanadeAffine(It, It1, rect)
    print('M ', M)

    # Transform the old rect to new one
    corners = np.array([[rect[0], rect[1], 1],
                        [rect[2], rect[3], 1]]).transpose()
    newRect = np.matmul(M, corners).transpose().reshape((4, ))
    rects.append(newRect)

    # Show image
    print("Plotting: ", rect)
    ax.add_patch(patches.Rectangle((rect[0], rect[1]), rect[2]-rect[0]+1, rect[3]-rect[1]+1, linewidth=2, edgecolor='red', fill=False))
    plt.imshow(It1, cmap='gray')
    save_path = "../results/lk_affine/%s/frame%06d.jpg" % (data_name, i+1)
    mkdir_if_missing(save_path)
    plt.savefig(save_path)
    if do_display:
        plt.pause(0.01)
    ax.clear()
