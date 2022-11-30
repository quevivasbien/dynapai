from . import dynapai as dp
import matplotlib.pyplot as plt

ylabels = ["$x_s$", "$x_p$", "$i_s$", "$i_p$"]

def plot(strategies, title = None, labels = None, logscale = True, figsize = None, show = True):
    assert isinstance(strategies, dp.Strategies) or isinstance(strategies, dp.InvestStrategies)
    data = strategies.data()
    labels = labels or [f"Player {i+1}" for i in range(data.shape[1])]
    n_axs = data.shape[2]
    figsize = figsize or (8, 2*n_axs)
    fig, axs = plt.subplots(n_axs // 2, 2, figsize = figsize, layout = "constrained")
    for j in range(n_axs):
        ax_idx = j if axs.ndim == 1 else (j // 2, j % 2)
        for i in range(data.shape[1]):
            axs[ax_idx].plot(data[:, i, j], label = labels[i])
        axs[ax_idx].set_xlabel("t")
        axs[ax_idx].set_ylabel(ylabels[j])
        axs[ax_idx].legend()
        if logscale:
            axs[ax_idx].set_yscale("log")
    if title is not None:
        fig.suptitle(title)
    if show:
        plt.show()
