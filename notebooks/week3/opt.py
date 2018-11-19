import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import math


def func_1(x):
    y = 2*np.sin(x) + x + 1
    return y

# 3-D example
def func_2(x, y):
    z= np.exp(-((x+1)**2 + (y+1)**2)) + 2*np.exp(-((x-1)**2 + (y-1)**2))
    return z

#Gradient ascent for func_1 
def gradient_ascent(previous_x, learning_rate, epoch):
    x_gd = []
    y_gd = []
    x_gd.append(previous_x)
    y_gd.append(func_1(previous_x))
    for i in range(epoch):
        current_x = previous_x + learning_rate*(2*np.cos(previous_x) + 1)
        x_gd.append(current_x)
        y_gd.append(func_1(current_x))
        previous_x = current_x

    return x_gd, y_gd

# Circle equation
def circle_points(radius, center):
    p1 = [center[0], center[1] - radius]
    p2 = [center[0] + radius, center[1]]
    p3 = [center[0] - radius, center[1]]
    p4 = [center[0], center[1]+radius]
    return [p1, p2, p3, p4]

# Hill climbing algorithm which returns a path of type list containing centroid 
# and corresponding surrounding points 
def hill_climbing(x0, learning_rate, epoch):
    i = 0
    centroid = x0
    f_c = func_2(centroid[0], centroid[1])
    r= 0.5
    path = []
    while i < epoch:
        points = circle_points(r, centroid)
        path.append([centroid, points])
        f = [func_2(item[0], item[1]) for item in points]
        ind = np.argmax(f)
        if max(f) >= f_c:
            centroid = [points[ind][0], points[ind][1]]
            f_c = func_2(centroid[0], centroid[1])
            r*= learning_rate
            i+=1
        else:
            i = epoch  
    x_hc =[]
    y_hc =[]
    z_hc = []
    for i in path:
        x_t = i[0][0]
        y_t = i[0][1]
        x_hc.append(x_t)
        y_hc.append(y_t)
        z_hc.append(func_2(x_t,y_t))
    return x_hc, y_hc, z_hc


# Plotting functions
def plot_base_func_1(x, y):
    fig, ax = plt.subplots(1, 1, sharey = True)
    ax.set_xlim([0, 10])
    ax.set_ylim([-3, 12.109627])
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    return fig, ax

def plot_base_func_2(x, y, z):    
    fig = plt.figure()
    
    ax = Axes3D(fig)
    surf = ax.plot_surface(x, y, z, edgecolor='none', rstride=1,
                        cstride=1, cmap='jet')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zlabel(r'$z$')

    return fig, ax, surf



def plot_func1(x, y):
    # Plot base figure
    fig, ax = plot_base_func_1(x,y)

    # Point
    #ax.plot([1], [func_1(1)], 'o', lw = 1.5, color = 'b')

    # First arrow 
    #ax.annotate('', xy=(1.8, 4.24), xytext=(1, func_1(1)),
      #                 arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
      #                 va='center', ha='center')

    # Second arrow (duality)
    #ax.annotate('', xy=(0.2, 3.12), xytext=(1, func_1(1)),
     #                  arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
      #                 va='center', ha='center')

    # Plot Function
    ax.plot(x, y, lw = 0.9, color = 'k')

    # Plot gradient ascent
    ''''ax.scatter(x_gd, y_gd, c = 'b')
    for i in range(1, epoch+1):
        ax.annotate('', xy=(x_gd[i], y_gd[i]), xytext=(x_gd[i-1], y_gd[i-1]),
                       arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
                       va='center', ha='center')'''

    # Arrow pointing to global maximum
    ''''ax.annotate('', xy=(8.3, 11), xytext=(9, 8),
                       arrowprops={'arrowstyle': '->', 'color': 'g', 'lw': 2},
                       va='center', ha='center')'''

    #plt.savefig(figs_path+'point_arrow.png')
    plt.show()
    
def plot_func2(x, y, z):
    fig1, ax1, surf = plot_base_func_2(x,y, z)
    # Plot the first centroid and surrounding points
    '''num = 1
    t = np.array(path[num][0])
    t_ = t[:, np.newaxis]
    ax1.plot(*t_, func_2(*t_), 'r*', markersize=5)
    for item in path[num][1]:
        t = np.array(item)
        t_ = t[:, np.newaxis]
        ax1.plot(*t_, func_2(*t_), 'y*', markersize=5)   ''' 

    # Plot all centroids
    ''''for m in path:
        t = np.array(m[0])
        t_ = t[:, np.newaxis]
        ax1.plot(*t_, func_2(*t_), 'r*', markersize=5)
        for item in m[1]:
            t = np.array(item)
            t_ = t[:, np.newaxis]
            ax1.plot(*t_, func_2(*t_), 'y*', markersize=5)  '''    


    #plt.savefig(figs_path+'3_d_plot_c2.png')
    
def animate_gd(x, y, x_gd, y_gd):
    plt.rcdefaults()
    # Plot base figure
    fig, ax = plot_base_func_1(x,y)
    ax.plot(x, y, lw = 0.9, color = 'k')

    # Define plot elements
    line, = ax.plot([], [], 'r', label = 'Gradient Ascent', lw = 1.5)
    point, = ax.plot([], [], 'bo')
    value_display = ax.text(0.02, 0.02, '', transform=ax.transAxes)

    # Initialize animation
    def init():
        line.set_data([], [])
        point.set_data([], [])
        value_display.set_text('')
        return line, point, value_display

    # Animate 
    def animate(i):
        line.set_data(x_gd[:i], y_gd[:i])
        point.set_data(x_gd[i], y_gd[i])
        value_display.set_text('Max = ' + str(y_gd[i]))
        return line, point, value_display

    ax.legend(loc = 2)
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(x_gd), interval=120, 
                                   repeat_delay=60, blit=True)
    plt.show()
    
def get_points(path):
    x_hc =[]
    y_hc =[]
    z_hc = []
    for i in path:
        x_t = i[0][0]
        y_t = i[0][1]
        x_hc.append(x_t)
        y_hc.append(y_t)
        z_hc.append(func_2(x_t,y_t))

    return x_hc, y_hc, z_hc