import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

fig = plt.figure()
fig.set_dpi(100)
fig.set_size_inches(7, 6.5)
ax = plt.axes(xlim=(0, 15), ylim=(0, 15))
circ = plt.Circle((5, -5), 0.75, fc='y')
rect = plt.Rectangle((0.5, 0.5), 1.5, 1.5, fc='r')
plt.axis('off')

def init():
    circ.center = (7, 6)
    rect.center = (1.5,1.5)
    ax.add_patch(circ)
    ax.add_patch(rect)
    return []

def animationManage(i,circ,rect):
    animate2D(i,circ)
    animate1D(i,rect)
    return []

def animate2D(i, patch):
    x, y = patch.center
    if i < 400:
        x = 7 + 1.5 * np.sin(np.radians(5*i))
        y = 6 + 1.5 * np.cos(np.radians(5*i))
    patch.center = (x, y)
    #print(patch.center)
    return patch,

def animate1D(i, patch):
    if i <= 120:
        patch.set_y(0.5 + 0.07*i)
    elif 120 < i <= 240:
        pass#patch.set_x(0.5 + 0.07*(i-120))
    elif 240 < i <= 360:
        patch.set_x(0.5 + 0.07*(i-240))#patch.set_y(8.9 - 0.07*(i-240))
    else:
        pass#patch.set_x(8.9 - 0.07*(i-360))
    return patch,


anim = animation.FuncAnimation(fig, animationManage, 
                               init_func=init, 
                               frames=480,
                               fargs=(circ,rect,), 
                               interval=15,
                               blit=False)


#plt.show()

anim.save('animation.mp4', fps=15, 
          extra_args=['-vcodec', 'h264', 
                      '-pix_fmt', 'yuv420p'])
