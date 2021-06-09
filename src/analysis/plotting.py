import matplotlib.pyplot as plt
import os
import seaborn as sns

plt.switch_backend('agg')

# plot default values
LINESTYLE = "-"
LINEWIDTH = 3
FONTSIZE = 16


def create_plot(data, data_name, logx=False, plot_errorbar=True,
                avg_time=0, performance_metric="BLEU"):
    global LINESTYLE, LINEWIDTH, FONTSIZE
    errorbar_kind = 'shade'
    errorbar_alpha = 0.1
    x_axis_time = avg_time != 0

    _, cur_ax = plt.subplots(1, 1)
    cur_ax.set_title(data_name, fontsize=FONTSIZE)
    cur_ax.set_ylabel("Expected validation " +
                      performance_metric, fontsize=FONTSIZE)

    # if x_axis_time:
    #     cur_ax.set_xlabel("Training duration", fontsize=FONTSIZE)
    # else:
    #     cur_ax.set_xlabel("Hyperparameter assignments", fontsize=FONTSIZE)

    if logx:
        cur_ax.set_xscale('log')

    means = data['mean']
    vars = data['var']
    max_acc = data['max']
    min_acc = data['min']

    if x_axis_time:
        x_axis = [avg_time * (i+1) for i in range(len(means))]
    else:
        x_axis = [i+1 for i in range(len(means))]

    if plot_errorbar:
        if errorbar_kind == 'shade':
            minus_vars = [
                x - y if (x - y) >= min_acc else min_acc for x, y in zip(means, vars)]
            plus_vars = [
                x + y if (x + y) <= max_acc else max_acc for x, y in zip(means, vars)]
            plt.fill_between(x_axis,
                             minus_vars,
                             plus_vars,
                             alpha=errorbar_alpha)
        else:
            cur_ax.errorbar(x_axis,
                            means,
                            yerr=vars,
                            linestyle=LINESTYLE,
                            linewidth=LINEWIDTH)
    cur_ax.plot(x_axis,
                means,
                linestyle=LINESTYLE,
                linewidth=LINEWIDTH)

    left, right = cur_ax.get_xlim()

    plt.xlim((left, right))
    plt.locator_params(axis='y', nbins=10)
    plt.tight_layout()

    save_plot(data_name, logx, plot_errorbar, avg_time)


def save_plot(data_name, logx, plot_errorbar, avg_time):
    name = "plots/{}_logx={}_errorbar={}_avgtime={}.pdf".format(
        data_name, logx, plot_errorbar, avg_time)

    if not os.path.exists("plots"):
        os.mkdir("plots")
    plt.savefig(name, dpi=300)


def create_prune_plot(data, title):
    global LINESTYLE, LINEWIDTH, FONTSIZE

    _, cur_ax = plt.subplots(1, 1)
    # cur_ax.set_title(title, fontsize=FONTSIZE)
    cur_ax.set_ylabel("Pruned runs ", fontsize=FONTSIZE)
    cur_ax.set_xlabel("Completed hyperparameter runs", fontsize=FONTSIZE)

    num_pruned = [trial_num - i for i, trial_num in enumerate(data)]

    cur_ax.plot([i+1 for i in range(len(data))],
                num_pruned,
                linestyle=LINESTYLE,
                linewidth=LINEWIDTH)

    left, right = cur_ax.get_xlim()

    plt.xlim((left, right))
    plt.locator_params(axis='y', nbins=10)
    plt.tight_layout()

    save_prune_plot(title)


def save_general_plot(data_name, plot_name):
    name = "plots/{}_{}.pdf".format(data_name, plot_name)

    if not os.path.exists("plots"):
        os.mkdir("plots")
    plt.savefig(name, dpi=300)


def save_prune_plot(data_name):
    save_general_plot(data_name, 'pruneplot')


def create_violin_plot(data_name, values):
    sns.violinplot(x=values)
    save_general_plot(data_name, 'violinplot')
