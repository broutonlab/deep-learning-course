import matplotlib.style as mplstyle
import numpy as np
from matplotlib import cm, rc
import torch


def optim_install_dependencies():
    import pip
    import importlib
    try:
        importlib.import_module("torch_optimizer")
        import torch_optimizer
        if torch_optimizer.Ranger is None:
            raise ImportError('Wrong torch_optim version!')
    except ImportError:
        import sys
        import subprocess
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', 'torch_optimizer==0.3.0'])


def optim_configure_notebooks():
    mplstyle.use('fast')
    rc('animation', html='jshtml')
    try:
        importlib.import_module("google.colab")
        IN_COLAB = True
    except:
        IN_COLAB = False

    if IN_COLAB:
        from google.colab import output
        output.enable_custom_widget_manager()


def print_convergence(path):
    print(f'{path["name"]} converged to: {path["xs"][-1]}, {path["ys"][-1]}')
    print(f'Last loss: {path["zs"][-1]}')
    print(f'Best loss: {path["best_loss"]}')


def path_from_trace(optimizer, trace):
    path_name = optimizer.__name__

    path = {
        "xs": trace[:, 0],
        "ys": trace[:, 1],
        "zs": trace[:, 2],
        "name": path_name,
        "best_loss": min(trace[:, 2])
    }

    return path


def run_optim(w, f, optimizer, n_iter, **optim_params):
    """
    Runs step-by-step optimization and outputs sequence of
    optimizer steps

    Parameters
    ----------
    w : array_like
        Start vector to optimize. Shouldn't be a torch.Tensor.
    f
        Loss function.
    optimizer
        Numpy-compatible function to calculate each step.
    n_iter : int
        Number of iterations.
    """

    trace = np.empty((n_iter + 1, 3))
    trace[0, :] = np.array([w[0], w[1], f(w)])

    for i in range(n_iter):
        # we're moving our array to torch
        # to avoid computing the gradient ourselves
        w_t = torch.tensor(w, requires_grad=True)
        loss = f(w_t)
        loss.backward()
        # prevent gradient from exploding
        torch.nn.utils.clip_grad_norm_(w_t, 100)
        # making it work for our numpy-based optimization
        dw = w_t.grad.detach().numpy()
        w, optim_params = optimizer(w, dw, **optim_params)
        trace[i+1, :] = np.array([w[0], w[1], loss.item()])

    return path_from_trace(optimizer, trace)


def run_torch_optim(w, f, optimizer_cls, n_iter, **optim_params):
    """
    Runs torch-based optimizer and outputs sequence of
    optimizer steps

    Parameters
    ----------
    w : array_like
        Start vector to optimize. Shouldn't be a torch.Tensor.
    f
        Loss function.
    optimizer_cls : torch.Optimizer
        Pytorch optimizer class.
    n_iter : int
        Number of iterations.
    """

    trace = np.empty((n_iter + 1, 3))
    trace[0, :] = np.array([w[0], w[1], f(w)])
    w_t = torch.tensor(w, requires_grad=True)
    optimizer = optimizer_cls([w_t], **optim_params)
    for i in range(n_iter):
        optimizer.zero_grad()
        loss = f(w_t)
        loss.backward()
        # prevent gradient from exploding
        torch.nn.utils.clip_grad_norm_(w_t, 100)
        optimizer.step()
        w = w_t.detach().numpy()
        trace[i+1, :] = np.array([w[0], w[1], loss.item()])

    return path_from_trace(optimizer_cls, trace)
