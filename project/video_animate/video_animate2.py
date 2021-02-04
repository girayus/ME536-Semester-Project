import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

fig = plt.figure()
fig.set_dpi(100)
fig.set_size_inches(7, 6.5)
ax = plt.axes(xlim=(0, 15), ylim=(0, 15))
trian = plt.Polygon(np.array([[2, 3], [1, 1], [3, 1]]), fc='g')
pen1 = plt.Polygon([[10.25, 1], [10, 2.15], [11, 3], [11, 1]], fc='b')
pen2 = plt.Polygon([[11, 1], [11, 3] , [12,2.15], [11.75, 1]], fc='g')
plt.axis('off')

def init():
    ax.add_patch(trian)
    ax.add_patch(pen1)
    ax.add_patch(pen2)
    return []

def animationManage(i,trian,pen1,pen2):
    animate2D(i,trian)
    animate1D(i,pen1)
    animate11D(i,pen2)
    return []

def animate2D(i, patch):
    ia = i%480
    if ia <= 160:
        patch.set_xy(np.array([[2, 3], [1, 1], [3, 1]]) + (0.05*ia)*np.array([[1, 1], [1, 1], [1, 1]]))
    elif 160 < ia <= 320:
        patch.set_xy(np.array([[10, 11], [9, 9], [11, 9]]) - (0.05*(ia-160))*np.array([[1, 0], [1, 0], [1, 0]]))
    else:
        patch.set_xy(np.array([[2, 11], [1, 9], [3, 9]]) - (0.05*(ia-320))*np.array([[0, 1], [0, 1], [0, 1]]))
    
    return patch,

def animate1D(i, patch):

    return patch,

def animate11D(i, patch):

    return patch,



anim = animation.FuncAnimation(fig, animationManage, 
                               init_func=init, 
                               frames=2400,
                               fargs=(trian,pen1,pen2), 
                               interval=15,
                               blit=False)


#plt.show()

anim.save('animation.mp4', fps=15, 
          extra_args=['-vcodec', 'h264', 
                      '-pix_fmt', 'yuv420p'])
