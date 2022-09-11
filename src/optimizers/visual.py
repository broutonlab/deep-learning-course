import numpy as np


def visualize_3d(f, paths=None, colors=None, xlim=None, elev=None, azim=None, anim_length=5, frames_per_sec=4):
    x1 = np.linspace(xlim[0], xlim[1])
    x2 = np.linspace(xlim[0], xlim[1])
    X1, X2 = np.meshgrid(x1, x2)
    
    fig = plt.figure(figsize = [8, 6])
    
    ax = plt.axes(projection='3d', computed_zorder=False)
    #plotlyplot = py.plot_mpl(fig,filename="mpl-complex-scatter")
    ax.plot_surface(X1, X2, f([X1, X2]), cmap='jet', alpha=0.8, zorder=0)
    ax.plot_wireframe(X1, X2, f([X1, X2]), rcount=7, ccount=7, alpha=0.5, zorder=1)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    function_name = f.__name__
    ax.set_zlabel(f'{function_name}(X, Y)')
    
    if paths is None:
        return
    
    #animate paths
    path_length = max(len(path["xs"]) for path in paths)
    lines = [ax.plot(path["xs"][0],
                           path["ys"][0],
                           path["zs"][0],
                           zorder=1,
                           alpha=0.9,
                           linewidth=3,
                           label=path["name"],
                           c=colors[i])[0] for i, path in enumerate(paths)]
    
    ax.legend()
    def animate(i):
        for path, line in zip(paths, lines):
            line.set_data([path["xs"][:i], path["ys"][:i]])
            line.set_3d_properties(path["zs"][:i])
            #scatter._offsets3d = (path["xs"][:i], path["ys"][:i], path["zs"][:i])

        ax.set_title(f"Iter: {i}")

        return lines
    
    frame_length = 1000/frames_per_sec
    num_frames = 1000*anim_length/frame_length
    range_step = int(path_length/num_frames)
    anim = FuncAnimation(fig, animate, frames=range(0, path_length, range_step), interval=frame_length, blit=True)
    plt.close()
    return anim



def plot_losses(paths, colors):
    fig = plt.figure(figsize = [8, 4])
    ax = plt.axes()
    ax.set_xlabel('Iteration number')
    ax.set_ylabel('Loss')
    ax.set_yticklabels([])
    path_length = max(len(path["zs"]) for path in paths)
    scatters = [ax.plot(range(path_length),
                           path["zs"],
                           label=path["name"],
                           c=colors[i]) for i, path in enumerate(paths)]
    ax.legend()