import os
import glob
from collections import defaultdict, namedtuple
from typing import DefaultDict, Dict, Union

import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np
import torch
from typing_extensions import List

Result = namedtuple('Result', ['cosines_checkpoint', 'test_losses_checkpoint', 'norms_checkpoint'])


def rec_dd():
    return defaultdict(rec_dd)


def fill_subplot(ax, title, xs, ys, xlabel, ylabel, xscale='linear', legend_labels: Union[str, List[str]] = "",
                 markersize=4, linestyle='solid'):
    if isinstance(ys[0], list):
        for i in range(len(ys)):
            if isinstance(xs[0], list):
                ax.plot(xs[i], ys[i], label=legend_labels[i].replace('_', ''), markersize=markersize, linestyle=linestyle)
            else:
                ax.plot(xs, ys[i], label=legend_labels[i].replace('_', ''), markersize=markersize, linestyle=linestyle)
    else:
        ax.plot(xs, ys, label=legend_labels.replace('_', ''), markersize=markersize, linestyle=linestyle)
    ax.set_title(title)
    if xlabel != '':
        ax.set_xlabel(xlabel)
    ax.set_xscale(xscale)
    # if xscale == 'log':
    #     ax.set_xticks(xs)
    #     ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    if ylabel != '':
        ax.set_ylabel(ylabel)
    if legend_labels != '':
        # ax.legend(bbox_to_anchor=(1.0, 1.0))
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
                  ncol=3, fancybox=True)


def plot_lr_ablation(lr_results_dict: Dict[str, Result], keys, batch_sizes, batch_sizes_to_plot=[4, 16, 64]):
    print(lr_results_dict.keys())
    ncols = 2
    plt.style.use('seaborn-v0_8-white')
    matplotlib.rcParams.update({'font.size': 16})
    matplotlib.rcParams.update({'legend.frameon': True})
    matplotlib.rcParams.update({'legend.fancybox': True})
    matplotlib.rcParams['axes.prop_cycle'] = cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#1f77b4', '#ff7f0e', '#2ca02c', '#1f77b4', '#ff7f0e', '#2ca02c', ])
    xlim_new = None
    for k in keys:
        fig, ax = plt.subplots(1, ncols, figsize=(8 * ncols, 8), dpi=300)
        linestyles = {4: 'solid', 16: 'dashed', 64: 'dashdot'}
        for b in batch_sizes_to_plot:
            for lr_scaling in lr_results_dict.keys():
                cosines, norms = lr_results_dict[lr_scaling].cosines_checkpoint, lr_results_dict[
                    lr_scaling].norms_checkpoint
                if lr_scaling == 'constant':
                    label = 'const'
                else:
                    label = lr_scaling
                fill_subplot(ax[0], 'Batch size vs cosine(weight delta w.r.t bs=1)',
                             list(cosines[k][batch_sizes.index(b)].keys()),
                             list(cosines[k][batch_sizes.index(b)].values()),
                             'Iterations', 'Cosine Sim', xscale='linear', legend_labels=f'{label} / BS{b}',
                             linestyle=linestyles[b])
                ax[0].set_ylim([0.1, 1.0])
                # ax[1].set_ylim([0.0, 400.0])
                # if xlim_new is None:
                #     xlim = ax[0].get_xlim()
                #     xlim_new = [xlim[0], int(xlim[1] * 1.5)]
                # ax[0].set_xlim(xlim_new)
                # ax[1].set_xlim(xlim_new)
                fill_subplot(ax[1], 'Batch size vs norm(weight delta)',
                             list(norms[k][batch_sizes.index(b)].keys()),
                             list(norms[k][batch_sizes.index(b)].values()),
                             'Iterations', 'Norm', xscale='linear', legend_labels=f'{label} / BS{b}',
                             linestyle=linestyles[b])

            # disable_momentum_str = '_disable momentum' if disable_momentum else ''
            # fig.suptitle(
            #     f'Scene: {scene_name}. Checkpoint {checkpoint_iter}. Rescale betas: {rescale_betas}{disable_momentum_str}. LR scaling: {lr_scaling}. Warmup: {warmup_epochs} epochs. IID {iid_sampling}. Params: {k}')
        fig.tight_layout()
        os.makedirs(os.path.join('paper_figures', 'garden'), exist_ok=True)
        fig_save_path = os.path.join('paper_figures', 'garden', 'garden_lr_scaling_ablation.pdf')
        fig.savefig(fig_save_path)
        if k == '_features_dc':
            fig.show()
        plt.close(fig)


def plot_betas_ablation(adjust_betas_lr_results_dict: Dict[bool, Dict[str, Result]], keys, batch_sizes, batch_sizes_to_plot=[4, 16, 64]):
    print(adjust_betas_lr_results_dict.keys())
    ncols = 2
    plt.style.use('seaborn-v0_8-white')
    matplotlib.rcParams.update({'legend.frameon': True})
    matplotlib.rcParams.update({'legend.fancybox': True})
    matplotlib.rcParams['axes.prop_cycle'] = cycler('color', ['#1f77b4', '#ff3333', '#1f77b4', '#ff3333', '#1f77b4', '#ff3333'])
    xlim_new = None
    for k in keys:
        fig, ax = plt.subplots(1, ncols, figsize=(8 * ncols, 8), dpi=300)
        linestyles = {4: 'solid', 16: 'dashed', 64: 'dashdot'}
        for b in batch_sizes_to_plot:
            for adjust_betas in [True, False]:
                if adjust_betas:
                    label = ''
                else:
                    label = 'Fixed β'
                cosines, norms = adjust_betas_lr_results_dict[adjust_betas]['sqrt'].cosines_checkpoint, adjust_betas_lr_results_dict[adjust_betas]['sqrt'].norms_checkpoint
                fill_subplot(ax[0], 'Batch size vs cosine(weight delta w.r.t bs=1)',
                             list(cosines[k][batch_sizes.index(b)].keys()),
                             list(cosines[k][batch_sizes.index(b)].values()),
                             'Iterations', 'Cosine Sim', xscale='linear', legend_labels=f'BS {b} {label}',
                             linestyle=linestyles[b])
                ax[0].set_ylim([0.1, 1.0])
                # ax[1].set_ylim([0.0, 400.0])
                # if xlim_new is None:
                #     xlim = ax[0].get_xlim()
                #     xlim_new = [xlim[0], int(xlim[1] * 1.4)]
                # ax[0].set_xlim(xlim_new)
                # ax[1].set_xlim(xlim_new)
                fill_subplot(ax[1], 'Batch size vs norm(weight delta)',
                             list(norms[k][batch_sizes.index(b)].keys()),
                             list(norms[k][batch_sizes.index(b)].values()),
                             'Iterations', 'Norm', xscale='linear', legend_labels=f'BS {b} {label}',
                             linestyle=linestyles[b])

            # disable_momentum_str = '_disable momentum' if disable_momentum else ''
            # fig.suptitle(
            #     f'Scene: {scene_name}. Checkpoint {checkpoint_iter}. Rescale betas: {rescale_betas}{disable_momentum_str}. LR scaling: {lr_scaling}. Warmup: {warmup_epochs} epochs. IID {iid_sampling}. Params: {k}')
        fig.tight_layout()
        os.makedirs(os.path.join('paper_figures', 'garden'), exist_ok=True)
        fig_save_path = os.path.join('paper_figures', 'garden', 'garden_momentum_ablation.pdf')
        fig.savefig(fig_save_path)
        if k == '_features_dc':
            fig.show()
        plt.close(fig)



def plot(pts_dir):
    cosines_checkpoint = []
    test_losses_checkpoint = []
    norms_checkpoint = []
    plot_batch_sizes = [1, 4, 8, 16, 32, 64]
    param_keys = ['_xyz', '_rotation', '_scaling', '_opacity', '_features_dc']
    for_iteration = rec_dd()
    for pt_file in glob.glob(os.path.join(pts_dir, '*.pt')):
        d = torch.load(pt_file)
        keys = list(d['keys'])
        assert keys == param_keys
        checkpoint_iter = d['checkpoint_iter']
        batch_sizes: list = d['batch_sizes']
        assert batch_sizes == plot_batch_sizes
        rescale_betas = d['rescale_betas']
        lr_scaling = d['lr_scaling']
        disable_momentum = d['disable_momentum']
        iid_sampling = d['iid_sampling']
        assert isinstance(d['cosines_checkpoint'], dict)
        assert isinstance(d['test_losses_checkpoint'], list)
        assert isinstance(d['norms_checkpoint'], dict)
        # cosines_checkpoint.append(d['cosines_checkpoint'])
        # test_losses_checkpoint.append(d['test_losses_checkpoint'])
        # norms_checkpoint.append(d['norms_checkpoint'])
        # for k, batch_size in enumerate(batch_sizes):
        for_iteration[checkpoint_iter][disable_momentum][iid_sampling][rescale_betas][lr_scaling] = (
            Result(d['cosines_checkpoint'], d['test_losses_checkpoint'], d['norms_checkpoint']))

    plot_lr_ablation(for_iteration[15000][False][False][True], param_keys, batch_sizes)
    plot_betas_ablation(for_iteration[15000][False][False], param_keys, batch_sizes)


if __name__ == '__main__':
    pts_dir = './plots_grad_delta_new/garden'
    plot(pts_dir)
