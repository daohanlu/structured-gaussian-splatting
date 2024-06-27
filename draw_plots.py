import os
import glob
from collections import defaultdict, namedtuple
from typing import DefaultDict, Dict, Union

import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np
import torch
# from typing_extensions import List
from typing import List

Result = namedtuple('Result', ['cosines_checkpoint', 'test_losses_checkpoint', 'norms_checkpoint'])


def rec_dd():
    return defaultdict(rec_dd)


def fill_subplot(ax, title, xs, ys, xlabel, ylabel, xscale='linear', legend_labels: Union[str, List[str]] = "",
                 markersize=4, linestyle='solid', **kwargs):
    linewidth = 3
    if isinstance(ys[0], list):
        for i in range(len(ys)):
            if isinstance(xs[0], list):
                ax.plot(xs[i], ys[i], label=legend_labels[i].replace('_', ''), markersize=markersize,
                        linestyle=linestyle, linewidth=linewidth, **kwargs)
            else:
                ax.plot(xs, ys[i], label=legend_labels[i].replace('_', ''), markersize=markersize, linestyle=linestyle,
                        linewidth=linewidth, **kwargs)
    else:
        ax.plot(xs, ys, label=legend_labels.replace('_', ''), markersize=markersize, linestyle=linestyle,
                linewidth=linewidth, **kwargs)
    ax.set_title(title, fontsize=24)
    if xlabel != '':
        ax.set_xlabel(xlabel, fontsize=22)
    ax.set_xscale(xscale)
    # if xscale == 'log':
    #     ax.set_xticks(xs)
    #     ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    if ylabel != '':
        ax.set_ylabel(ylabel, fontsize=22)
    # if legend_labels != '':
    #     # ax.legend(bbox_to_anchor=(1.0, 1.0))
    #     ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
    #               ncol=3, fancybox=True, shadow=True)


red = '#ff3333'
blue = '#1f77b4'
green = '#2ca02c'
orange = '#ff7f0e'


def plot_lr_ablation(lr_results_dict: Dict[str, Result], keys, batch_sizes, batch_sizes_to_plot=[4, 16, 64]):
    batch_sizes_to_plot = [4, 16, 32]
    print(lr_results_dict.keys())
    ncols = 2
    plt.style.use('seaborn-v0_8-white')
    matplotlib.rcParams.update({'font.size': 16})
    matplotlib.rcParams.update({'legend.frameon': True})
    matplotlib.rcParams.update({'legend.fancybox': True})
    # matplotlib.rcParams['axes.prop_cycle'] = cycler('color',
    #                                                 ['#1f77b4', '#ff7f0e', '#2ca02c', '#1f77b4', '#ff7f0e', '#2ca02c',
    #                                                  '#1f77b4', '#ff7f0e', '#2ca02c', ])
    xlim_new = None
    for k in keys:
        if k != "_features_dc":
            continue
        fig, ax = plt.subplots(1, ncols, figsize=(8 * ncols, 8), dpi=300)
        # fig, ax = plt.subplots(1, ncols, dpi=300)
        linestyles = {4: 'solid', 16: 'dashed', 32: 'dashdot', 64: 'dashdot'}
        for b in batch_sizes_to_plot:
            for lr_scaling in lr_results_dict.keys():
                cosines, norms = lr_results_dict[lr_scaling].cosines_checkpoint, lr_results_dict[
                    lr_scaling].norms_checkpoint
                if lr_scaling == 'constant':
                    label = 'const'
                else:
                    label = lr_scaling
                if label == 'const':
                    color = orange
                elif label == 'sqrt':
                    color = red
                elif label == 'linear':
                    color = blue
                fill_subplot(ax[0], 'Cumulative Update Direction',
                             list(cosines[k][batch_sizes.index(b)].keys()),
                             list(cosines[k][batch_sizes.index(b)].values()),
                             'Train Images', 'Cosine Sim (w/ BS 1)', xscale='linear', legend_labels=f'BS={b} {label}',
                             linestyle=linestyles[b], color=color)
                ax[0].set_ylim([0.1, 1.0])
                # ax[1].set_ylim([0.0, 400.0])
                # if xlim_new is None:
                #     xlim = ax[0].get_xlim()
                #     xlim_new = [xlim[0], int(xlim[1] * 1.5)]
                # ax[0].set_xlim(xlim_new)
                # ax[1].set_xlim(xlim_new)
                norm_diff = []
                for itr in norms[k][batch_sizes.index(b)].keys():
                    norm_diff.append(norms[k][batch_sizes.index(b)][itr] / norms[k][batch_sizes.index(1)][itr])
                fill_subplot(ax[1], 'Cumulative Update Magnitude',
                             list(norms[k][batch_sizes.index(b)].keys()),
                             norm_diff,
                             'Train Images', 'Norm Ratio (to BS 1)', xscale='linear', legend_labels=f'BS={b} {label}',
                             linestyle=linestyles[b], color=color)
                ax[1].set_ylim([0.0, 5.0])
                # ax[1].text(0.95, 0.95, 'BS=64 is out of the image field', transform=ax[1].transAxes, fontsize=18, verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

            # Adjust the bottom margin to make space for the legend
            fig.tight_layout()
            fig.subplots_adjust(bottom=0.25)

            # Add the legend to the bottom center after all plots
            handles, labels = ax[1].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.0), ncol=3, fancybox=True,
                       shadow=True, fontsize=18)

            # disable_momentum_str = '_disable momentum' if disable_momentum else ''
            # fig.suptitle(
            #     f'Scene: {scene_name}. Checkpoint {checkpoint_iter}. Rescale betas: {rescale_betas}{disable_momentum_str}. LR scaling: {lr_scaling}. Warmup: {warmup_epochs} epochs. IID {iid_sampling}. Params: {k}')
        # fig.tight_layout()
        os.makedirs(os.path.join('paper_figures', 'rubble'), exist_ok=True)
        fig_save_path = os.path.join('paper_figures', 'rubble', 'rubble_lr_scaling_ablation.pdf')
        fig.savefig(fig_save_path)
        fig.savefig(fig_save_path.replace('.pdf', '.png'))
        # if k == '_features_dc':
        # fig.show()
        plt.close(fig)


def plot_betas_ablation(adjust_betas_lr_results_dict: Dict[bool, Dict[str, Result]], keys, batch_sizes,
                        batch_sizes_to_plot=[4, 16, 64]):
    batch_sizes_to_plot = [4, 16, 32]
    print(adjust_betas_lr_results_dict.keys())
    ncols = 2
    plt.style.use('seaborn-v0_8-white')
    matplotlib.rcParams.update({'legend.frameon': True})
    matplotlib.rcParams.update({'legend.fancybox': True})
    # matplotlib.rcParams['axes.prop_cycle'] = cycler('color',
    #                                                 ['#1f77b4', '#ff3333', '#1f77b4', '#ff3333', '#1f77b4', '#ff3333'])
    xlim_new = None
    for k in keys:
        fig, ax = plt.subplots(1, ncols, figsize=(8 * ncols, 8), dpi=300)
        linestyles = {4: 'solid', 16: 'dashed', 32: 'dashdot', 64: 'dashdot'}
        for b in batch_sizes_to_plot:
            for adjust_betas in [False, True]:
                if adjust_betas:
                    label = 'adjust β'
                else:
                    label = 'const. β'
                if adjust_betas:
                    color = red
                else:
                    color = blue
                cosines, norms = adjust_betas_lr_results_dict[adjust_betas]['sqrt'].cosines_checkpoint, \
                adjust_betas_lr_results_dict[adjust_betas]['sqrt'].norms_checkpoint
                fill_subplot(ax[0], 'Cumulative Update Direction',
                             list(cosines[k][batch_sizes.index(b)].keys()),
                             list(cosines[k][batch_sizes.index(b)].values()),
                             'Train Images', 'Cosine Sim. (w/ BS 1)', xscale='linear', legend_labels=f'BS={b} {label}',
                             linestyle=linestyles[b], color=color)
                ax[0].set_ylim([0.1, 1.0])
                # ax[1].set_ylim([0.0, 400.0])
                # if xlim_new is None:
                #     xlim = ax[0].get_xlim()
                #     xlim_new = [xlim[0], int(xlim[1] * 1.4)]
                # ax[0].set_xlim(xlim_new)
                # ax[1].set_xlim(xlim_new)
                norm_diff = []
                for itr in norms[k][batch_sizes.index(b)].keys():
                    norm_diff.append(norms[k][batch_sizes.index(b)][itr] / norms[k][batch_sizes.index(1)][itr])
                fill_subplot(ax[1], 'Cumulative Update Magnitude',
                             list(norms[k][batch_sizes.index(b)].keys()),
                             norm_diff,
                             'Train Images', 'Norm Ratio (to BS 1)', xscale='linear', legend_labels=f'BS={b} {label}',
                             linestyle=linestyles[b], color=color)
                ax[1].set_ylim([0.0, 2.0])

            # Adjust the bottom margin to make space for the legend
            fig.tight_layout()
            fig.subplots_adjust(bottom=0.20)

            # Add the legend to the bottom center after all plots
            handles, labels = ax[1].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.0), ncol=3, fancybox=True,
                       shadow=True, fontsize=18)

            # disable_momentum_str = '_disable momentum' if disable_momentum else ''
            # fig.suptitle(
            #     f'Scene: {scene_name}. Checkpoint {checkpoint_iter}. Rescale betas: {rescale_betas}{disable_momentum_str}. LR scaling: {lr_scaling}. Warmup: {warmup_epochs} epochs. IID {iid_sampling}. Params: {k}')
        # fig.tight_layout()
        os.makedirs(os.path.join('paper_figures', 'rubble'), exist_ok=True)
        fig_save_path = os.path.join('paper_figures', 'rubble', 'rubble_momentum_ablation.pdf')
        fig.savefig(fig_save_path)
        fig.savefig(fig_save_path.replace('.pdf', '.png'))
        # if k == '_features_dc':
        #     fig.show()
        plt.close(fig)


def plot_cosine_similarity(pts_dir):
    matplotlib.rcParams.update({'legend.frameon': True})
    matplotlib.rcParams.update({'legend.fancybox': True})
    max_batch_size = 64
    for pt_file in glob.glob(os.path.join(pts_dir, '*.pt')):
        d = torch.load(pt_file)
        scene_name = d['scene_name']
        cosines = d['cosines_checkpoint']
        sparsities = d['sparsities_checkpoint']
        variances = d['variances_checkpoint']
        SNRs = d['SNRs_checkpoint']
        keys = d['keys']
        num_trials = d['num_trails']
        iters_list = d['iters_list']
        sampling = d['sampling']

        scene_name = 'rubble'
        for k in keys:
            ncols = 3
            fig, ax = plt.subplots(1, ncols, figsize=(8 * ncols, 8), dpi=300)
            for i, iter in enumerate(iters_list):
                xs = np.arange(1, len(variances[i][k]) + 1)
                fill_subplot(ax[0], 'Batch size vs Grad Sparsity', xs[:max_batch_size],
                             sparsities[i][k][:max_batch_size],
                             'Batch size (log scale)', 'Sparsity', xscale='log', legend_labels='iter ' + str(iter))
                fill_subplot(ax[1], 'Batch size vs Grad Variance', xs[:max_batch_size],
                             variances[i][k][:max_batch_size],
                             'Batch size', 'Avg Parameter Variance', xscale='linear', legend_labels='iter ' + str(iter))
                fill_subplot(ax[2], 'Batch size vs Grad Precision', xs[:max_batch_size],
                             1 / variances[i][k][:max_batch_size],
                             'Batch size', '1/(Avg Parameter Variance)', xscale='linear', legend_labels='iter ' + str(iter))
            # fig.suptitle(f'Scene: {scene_name}. Param group: {k.replace("_", "")}.')
            # fig.tight_layout()

            # Adjust the bottom margin to make space for the legend
            fig.tight_layout()
            fig.subplots_adjust(bottom=0.15)

            # Add the legend to the bottom center after all plots
            handles, labels = ax[1].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.0), ncol=3, fancybox=True,
                       shadow=True, fontsize=18)

            os.makedirs(os.path.join('plots_snr', scene_name), exist_ok=True)
            fig_save_path = os.path.join('plots_snr', scene_name,
                                         f'scene_{scene_name}_param_{k.replace("_", "")}_sampling_{sampling}_trials_{num_trials}.pdf')
            fig.savefig(fig_save_path)
            # fig.show()
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
    pts_dir = './plots_grad_delta_new/rubble'
    plot(pts_dir)
    plot_cosine_similarity('./plots_snr/rubble-full-batch')
