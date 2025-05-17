import matplotlib.pyplot as plt
import numpy as np
import sys
from matplotlib.colors import LogNorm

DEFAULT_GRAPH_PATH = "confusion_matrix.npy"

path = DEFAULT_GRAPH_PATH
if len(sys.argv)==2:
    path = sys.argv[1]

dat = np.load(path)

k = dat.shape

if(k[0]!=k[1]):
    print("THERE IS AN ERROR AT DAT FILE")
    sys.exit(-1)

nlabel = []
for i in range(k[0]):
    nlabel.append(str(i+1))

crtsum = 0
errsum = 0

for i in range(k[0]):
    for j in range(k[0]):
        if i!=j:
            errsum += dat[i][j]
        else:
            crtsum += dat[i][j]
        
        dat[i][j] +=1

evsum = errsum+crtsum
corr = crtsum/evsum

mx = max(dat.flatten())
threshold = mx*0.80
fig, ax = plt.subplots(figsize=(9,7))

nnorm = None
if mx>100:
    nnorm = LogNorm(vmin=1, vmax=mx)
    threshold = mx*0.25

im = ax.imshow(dat, norm=nnorm,cmap = plt.cm.viridis)

cbar = ax.figure.colorbar(im,ax=ax)



ax.set_xticks(range(k[0]),labels=nlabel,rotation=0)
ax.set_yticks(range(k[0]),labels=nlabel,rotation=0)

ax.xaxis.tick_top()

for i in range(k[0]):
    for j in range(k[0]):
        clr = "b"
        if dat[i][j]<threshold:
            clr = "w"
        text = ax.text(j,i,dat[i][j]-1,ha="center",va="center",color=clr)


ax.set_title('Confusion Matrix')
txtLine = f"{'DATA SUM = ':<15}{evsum:<10}{'Correct(%) = ':<15}{corr*100:<10.2f}{'ErrSUM = ':<15}{errsum:<10}{'CorSUM = ':<15}{crtsum:<10}"
plt.figtext(0.5,0.01,txtLine,horizontalalignment='center',fontsize=10)

fig.tight_layout()
plt.show()