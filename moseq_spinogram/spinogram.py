from argparse import Namespace
import logging
from typing import List, Optional
from typing_extensions import TypedDict
import matplotlib as mpl

from moseq_spinogram.slice import Slice, load_h5_timestamps, prep_slice_data
from moseq_spinogram.util import NumpyEncoder
mpl.use('Agg')

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, wait

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance as ssd
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import ImageGrid

from moseq2_viz.model.dist import get_behavioral_distance
from moseq2_viz.util import parse_index
from scipy import signal
from scipy.cluster import hierarchy
from tqdm import tqdm

class SpinogramTimepoint(TypedDict):
    x: np.ndarray
    y: np.ndarray
    t: float
    a: float  # alpha value for plotting

class SpinogramData(TypedDict):
    data: List[SpinogramTimepoint]
    sid_raw: int
    sid_usage: int
    sid_frames: int
    example_no: int



def save_figure(fig: plt.Figure, basename: str) -> None:
    """Save a matplotlib figure to PDF and PNG formats.
    Args:
        fig (plt.Figure): The matplotlib figure to save.
        basename (str): The base name for the output files.
    """
    fig.savefig("{}.pdf".format(basename))
    fig.savefig("{}.png".format(basename))
    sys.stderr.write("Saved plot output to {}.[png|pdf]\n".format(basename))
#end save_figure()

def save_data(data: SpinogramData, basename: str):
    with open("{}.json".format(basename), 'w+') as d_out:
        json.dump(data, d_out, cls=NumpyEncoder)
#end save_data()


def plot_one_single_syllable(args: Namespace):
    if args.no_plot:
        axes = None
    else:
        fig, axes = plt.subplots(nrows=1, ncols=1, squeeze=False)
        axes = axes.flatten()

    data = produce_spinograms([args.syllable_id], 1, axes, args)

    basename = "{}.single-raw{}-usage{}-frames{}".format(args.name,
                                     args.label_map[args.syllable_id]['raw'],
                                     args.label_map[args.syllable_id]['usage'],
                                     args.label_map[args.syllable_id]['frames'])

    if args.save_data:
        save_data(data, basename)

    if not args.no_plot:
        save_figure(fig, basename)
    else:
        sys.stderr.write("Skipping plot output because --no-plot was passed.\n")
#end plot_one_single_syllable()

def plot_many_single_syllable(args: Namespace):
    if args.max_examples <= 5:
        ncols = args.max_examples
        nrows = 1
    else:
        ncols = 5
        nrows = int(np.ceil(args.max_examples // ncols))

    if args.no_plot:
        axes = None
    else:
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False, sharex=True, sharey=True, figsize=(3*ncols, 2*nrows))
        axes = axes.flatten()

    all_data = produce_spinograms([args.syllable_id], args.max_examples, axes, args)

    basename = "{}.multi-raw{}-usage{}-frames{}".format(args.name,
                                     args.label_map[args.syllable_id]['raw'],
                                     args.label_map[args.syllable_id]['usage'],
                                     args.label_map[args.syllable_id]['frames'])

    if args.save_data:
        save_data(all_data, basename)

    if axes is not None:
        # style the axes
        for i, ax in enumerate(axes):
            if i >= len(all_data):
                ax.axis('off')
            ax.set_xlabel(None)
            ax.set_ylabel(None)
            ax.set_title(None)

        fig.text(0.5, 0.04, 'Relative Lateral Position (mm)', ha='center')
        fig.text(0.04, 0.5, 'Height (mm)', va='center', rotation='vertical')
        fig.tight_layout()
        save_figure(fig, basename)
    else:
        sys.stderr.write("Skipping plot output because --no-plot was passed.\n")
#end plot_many_single_syllable

def plot_syllable_corpus(args: Namespace):
    if args.separate:
        plot_syllable_corpus_separate(args)
        return

    num_plots = args.max_syllable * args.max_examples

    if args.no_plot:
        axes = None
    else:
        ncols = 10
        fig, axes = plt.subplots(nrows=int(np.ceil(num_plots/ncols)), ncols=ncols, squeeze=False, sharex=True, sharey=True)
        axes = axes.flatten()

    all_data = produce_spinograms(np.arange(0, args.max_syllable), args.max_examples, axes, args)

    basename = "{}.corpus-{}-{}".format(args.name,
                                        'sorted' if args.sort else 'unsorted',
                                        args.count)

    if args.save_data:
        save_data(all_data, basename)

    if axes is not None:
        # style the axes
        for i, ax in enumerate(axes):
            if i >= len(all_data):
                ax.axis('off')
            ax.set_xlabel(None)
            ax.set_ylabel(None)

        #fig.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.2, wspace=0.1)
        fig.text(0.5, 0.04, 'Relative Lateral Position (mm)', ha='center')
        fig.text(0.04, 0.5, 'Height (mm)', va='center', rotation='vertical')
        fig.tight_layout()
        save_figure(fig, basename)
    else:
        sys.stderr.write("Skipping plot output because --no-plot was passed.\n")
#end plot_syllable_corpus()

def plot_syllable_corpus_separate(args: Namespace):
    for sid in tqdm(range(args.max_syllable)):
        if args.no_plot:
            axes = None
        else:
            fig, axes = plt.subplots(nrows=1, ncols=1, squeeze=False)

        data = produce_spinograms([sid], 1, axes, args)

        basename = "{}.single-raw{}-usage{}-frames{}".format(args.name,
                                        args.label_map[sid]['raw'],
                                        args.label_map[sid]['usage'],
                                        args.label_map[sid]['frames'])
        if args.save_data:
            save_data(data, basename)

        if not args.no_plot:
            save_figure(fig, basename)
            plt.close(fig)
        else:
            sys.stderr.write("Skipping plot output because --no-plot was passed.\n")

    if args.no_plot:
        sys.stderr.write("Skipped plot output because --no-plot was passed.\n")
#end plot_syllable_corpus_separate()

def plot_clustered_corpus(args: Namespace):
    _, sorted_index = parse_index(args.index)
    slice_gen = prep_slice_data(args.model, args.index, sort_labels=args.sort, count=args.count)

    sys.stderr.write("Computing behavioral distances...\n")
    dist = get_behavioral_distance(sorted_index,
                                   args.model,
                                   max_syllable=args.max_syllable,
                                   sort_labels_by_usage=args.sort,
                                   count=args.count,
                                   distances=[args.distance])
    Z = hierarchy.linkage(ssd.squareform(dist[args.distance]), args.linkage, optimal_ordering=True)

    if args.show_pca:
        pca_h5 = h5py.File(os.path.join(os.path.dirname(args.index), sorted_index['pca_path']), 'r')
        ncols = 3
        width_ratios = [6,3,2]
        width = 16
    else:
        ncols = 2
        width_ratios = [3,1]
        width = 12

    gs = GridSpec(args.max_syllable, ncols, width_ratios=width_ratios)
    fig = plt.figure(figsize=(width, 2*args.max_syllable))

    ax = fig.add_subplot(gs[:, 0])
    dendro = hierarchy.dendrogram(Z, ax=ax, orientation='left')
    ax.tick_params(axis='y', which='major', labelsize=48)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    shared_ax = None
    for i, sid in enumerate(tqdm(list(reversed(dendro['ivl'])))):
        sid = int(sid)
        if sid > args.max_syllable:
            break

        slices = slice_gen(sid, args.pick)

        ax = fig.add_subplot(gs[i, 1], sharex=shared_ax, sharey=shared_ax)
        if shared_ax is None:
            shared_ax = ax

        if len(slices) == 0:
            tqdm.write("No slices found for syllable {}".format(sid))
            ax.axis('off')
            continue
        # create_spinogram(sid, slices[0], args.color, numpts=args.num_samples, ax=ax)
        data = create_spinogram_data(slices[0], numpts=args.num_samples)
        create_spinogram_plot(data, args.color, ax=ax)
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.set_title(None)

        if args.show_pca:
            ax = fig.add_subplot(gs[i, 2])
            pca_path = '/scores/{}'.format(slices[0][1])
            s, e = slices[0][0]
            pca_data = pca_h5[pca_path][s:e, 0:10]
            x = np.arange(pca_data.shape[0])
            for i in range(pca_data.shape[1]):
                ax.plot(x, pca_data[:,i], label="PC{}".format(i))
            #ax.legend()

    basename = "{}.clustered-{}-{}-d_{}-l_{}".format(args.name,
                                                    'sorted' if args.sort else 'unsorted',
                                                    args.count,
                                                    args.distance,
                                                    args.linkage)

    if args.save_data:
        sys.stderr.write('WARNING: --save-data is not supported for option "plot-clustered". Skipping...\n')

    if not args.no_plot:
        #fig.text(0.5, 0.04, 'Relative Lateral Position (mm)', ha='center')
        #fig.text(0.04, 0.5, 'Height (mm)', va='center', rotation='vertical')
        fig.tight_layout()
        fig.savefig("{}.pdf".format(basename))
        fig.savefig("{}.png".format(basename))
        sys.stderr.write("Saved plot output to {}.[png|pdf]\n".format(basename))
    else:
        sys.stderr.write("Skipping plot output because --no-plot was passed.\n")
#end plot_clustered_corpus()


def produce_spinograms(syllables: List[int], examples: int, axes, args: Namespace):
    slice_gen = prep_slice_data(args.model, args.index, sort_labels=args.sort, count=args.count)
    all_data = []
    axes_counter = 0
    plot_map = {}

    if args.style == 'spinogram':
        method = partial_produce_spinograms
    elif args.style == 'point-cloud':
        method = partial_produce_point_clouds
    else:
        raise ValueError(f'Unexpected style "{args.style}"; expected either "spinogram" or "point-cloud"!')

    with ProcessPoolExecutor(max_workers=args.processors) as pool:
        for sid in tqdm(syllables, desc="Syllables"):
            slices = slice_gen(sid, args.pick)
            num_slices = len(slices)
            slice_iter = iter(slices)

            if num_slices == 0:
                tqdm.write("No slices found for syllable {}".format(sid))
                continue

            with tqdm(total=examples, desc="Examples", leave=False, disable=True) as pbar:

                futures_map = {}
                all_futures = []

                def queue_item(*call_args):
                    future = pool.submit(method, *call_args)
                    all_futures.append(future)
                    futures_map[future] = list(call_args)
                    future.add_done_callback(done_callback)
                #end queue_item()

                def done_callback(_f):
                    try:
                        _self_args = futures_map[_f]
                        _result = _f.result()
                        all_data.append(_result)

                        ax_id = _self_args[3]
                        if ax_id is not None:
                            plot_map[ax_id] = _result

                        pbar.update(1)
                    except Exception as e:
                        # print out the error in a nice way
                        fmt_args = [e, _self_args[1], _self_args[2] + 1]
                        tqdm.write("Something happened: {}; attempting to pick another slice for syllable {} example #{}".format(*fmt_args))
                        logging.error(e, exc_info=True)

                        # requeue the work item, BUT picking another slice
                        _self_args[-1] = next(slice_iter)
                        queue_item(*_self_args)
                    finally:
                        futures_map.pop(_f, None)
                #end done_callback()

                try:
                    for ex in range(examples):
                        try:
                            s = next(slice_iter)
                        except StopIteration:
                            tqdm.write("Not enough slices found for syllable {}, skipping example #{}".format(sid, ex + 1))
                            break

                        if axes is not None:
                            ax_id = axes_counter
                            axes_counter += 1
                            # try:
                            #     ax = next(axes_iter)
                            # except StopIteration:
                            #     tqdm.write(("Ran out of axes while on syllable {} example #{}.\n"
                            #         + "{} axes supplied; {} spinograms produced. Bailing out!").format(sid, ex, len(axes), len(all_data)))
                            #     break
                        else:
                            ax_id = None
                        queue_item(args, sid, ex, ax_id, s)

                finally:
                    # https://stackoverflow.com/questions/38258774/python-3-how-to-properly-add-new-futures-to-a-list-while-already-waiting-upon-i
                    # wait, while allowing for possible additions of jobs to the pool
                    while all_futures:
                        fs = all_futures[:]
                        for f in fs:
                            all_futures.remove(f)
                        wait(fs)

    for ax_id in tqdm(plot_map.keys(), desc="Plotting Data"):
        if args.style == 'spinogram':
            create_spinogram_plot(plot_map[ax_id], args.color, ax=axes[ax_id])

        elif args.style == 'point-cloud':
            create_point_cloud_plot(plot_map[ax_id], args.color, ax=axes[ax_id])

        else:
            raise ValueError(f'Unexpected style "{args.style}"; expected either "spinogram" or "point-cloud"!')


    return all_data
#end produce_spinograms()

def partial_produce_spinograms(args: Namespace, sid: int, ex: int, ax_id: int, slice: Slice) -> SpinogramData:
    try:
        data = create_spinogram_data(slice, numpts=args.num_samples)

        # decorate with additional metadata
        data['sid_raw'] = args.label_map[sid]['raw']
        data['sid_usage'] = args.label_map[sid]['usage']
        data['sid_frames'] = args.label_map[sid]['frames']
        data['example_no'] = ex

        return data
    except Exception as e:
        raise Exception("{} ({})".format(e, sys.exc_info()[2].tb_lineno))  # type: ignore[union-attr]
#end partial_produce_spinograms()


def partial_produce_point_clouds(args: Namespace, sid: int, ex: int, ax_id: int, slice: Slice):
    try:
        data = create_point_cloud_data(slice, numpts=args.num_samples)

        # decorate with additional metadata
        data['sid_raw'] = args.label_map[sid]['raw']
        data['sid_usage'] = args.label_map[sid]['usage']
        data['sid_frames'] = args.label_map[sid]['frames']
        data['example_no'] = ex

        return data
    except Exception as e:
        raise Exception("{} ({})".format(e, sys.exc_info()[2].tb_lineno))  # type: ignore[union-attr]
#end partial_produce_point_clouds()


def create_spinogram_data(slice: Slice, numpts: int = 10) -> SpinogramData:
    data: SpinogramData = {
        'data': [],
        'sid_raw': -1,
        'sid_usage': -1,
        'sid_frames': -1,
        'example_no': -1
    }
    with h5py.File(slice[2], 'r') as h5:
        frame_shape = h5['/frames'][()].shape
        num_frames = slice[0][1] - slice[0][0]
        all_timestamps = load_h5_timestamps(h5)
        t_init = all_timestamps[slice[0][0]]

        for t in range(numpts+1):
            offset = int(np.floor((t / numpts) * num_frames))
            frame_id = slice[0][0] + offset

            if frame_id >= frame_shape[0]:
                tqdm.write("Frame id {} out of range {}. skipping...\n".format(frame_id, frame_shape[0]))
                continue

            frame = h5['/frames'][frame_id]
            mask = h5['/frames_mask'][frame_id]
            y_data = get_midline_data(frame * mask)

            x = h5['/scalars/centroid_x_mm'][slice[0][0]:frame_id]
            y = h5['/scalars/centroid_x_mm'][slice[0][0]:frame_id]
            lmm = h5['/scalars/length_mm'][frame_id]
            lpx = h5['/scalars/length_px'][frame_id]
            dist = get_distance_traveled(x, y)
            t_elapsed = all_timestamps[frame_id] - t_init

            x_axis = dist + (np.arange(y_data.shape[0]) * (lmm / lpx))
            y_axis = y_data
            alpha = (t+1)/(numpts+1)

            data['data'].append({
                'x': x_axis,
                'y': y_axis,
                't': t_elapsed,
                'a': alpha
            })
    return data
#end create_spinogram_data()


def create_point_cloud_data(slice: Slice, numpts: int = 10):
    data: SpinogramData = {
        'data': [],
        'sid_raw': -1,
        'sid_usage': -1,
        'sid_frames': -1,
        'example_no': -1
    }
    with h5py.File(slice[2], 'r') as h5:
        frame_shape = h5['/frames'][()].shape
        num_frames = slice[0][1] - slice[0][0]
        all_timestamps = load_h5_timestamps(h5)
        t_init = all_timestamps[slice[0][0]]

        for t in range(numpts+1):
            offset = int(np.floor((t / numpts) * num_frames))
            frame_id = slice[0][0] + offset

            if frame_id >= frame_shape[0]:
                tqdm.write("Frame id {} out of range {}. skipping...\n".format(frame_id, frame_shape[0]))
                continue

            # calculate distance traveled since the start of the syllable
            x = h5['/scalars/centroid_x_mm'][slice[0][0]:frame_id]
            y = h5['/scalars/centroid_y_mm'][slice[0][0]:frame_id]
            lmm = h5['/scalars/length_mm'][frame_id]
            lpx = h5['/scalars/length_px'][frame_id]
            dist = get_distance_traveled(x, y)
            t_elapsed = all_timestamps[frame_id] - t_init

            frame = h5['/frames'][frame_id]
            mask = h5['/frames_mask'][frame_id]

            x_axis = []
            y_axis = []
            for y, x in np.argwhere(mask > 0):
                if frame[y, x] > 0:
                    x_axis.append((x * (lmm / lpx)) + dist)
                    y_axis.append(frame[y, x])

            alpha = (t+1)/(numpts+1)

            data['data'].append({
                'x': x_axis,
                'y': y_axis,
                't': t_elapsed,
                'a': alpha
            })
    return data
#end create_point_cloud_data()


def create_spinogram_plot(data: SpinogramData, color, ax: plt.Axes = None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    for t in data['data']:
        ax.plot(t['x'], t['y'], linewidth=2, color=color, alpha=t['a'])

    ax.set_ylim(0, 100)
    ax.set_xlim(0, 200)
    ax.set_aspect('equal')
    ax.set_xlabel('Relative Lateral Position (mm)')
    ax.set_ylabel('Height (mm)')
    ax.set_title('Syllable {}r/{}u/{}f, example {}'.format(data['sid_raw'],
                                                           data['sid_usage'],
                                                           data['sid_frames'],
                                                           data['example_no']))

    return ax
#end create_spinogram_plot()

def create_point_cloud_plot(data, color, ax: Optional[plt.Axes]=None):
    if ax is None:
        fig, axs = plt.subplots(len(data['data']), 1, sharex=True, sharey=True)
        
        axs = axs.flatten()
    else:
        fig = ax.get_figure()
        n_rows = len(data['data']) // 5
        n_cols = len(data['data']) // 2
        grid = ImageGrid(fig, (1,1,1), (n_rows, n_cols), axes_pad=(0.1, 0.5), cbar_mode=None)
        for cbax in grid.cbar_axes: fig._axstack.remove(cbax)
        fig.delaxes(ax)


    for i, (ax, t) in enumerate(zip(grid, data['data'])):
        ax.scatter(t['x'], t['y'], color=color, s=0.001, alpha=0.3)
        ax.set_title(f't={t["t"]:.0f}ms')
        ax.set_aspect('equal')
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 100)
        if i == 0:
            ax.set_xlabel('Relative Lateral Position (mm)')
            ax.set_ylabel('Height (mm)')

    fig.suptitle('Syllable {}r/{}u/{}f, example {}'.format(data['sid_raw'],
                                                           data['sid_usage'],
                                                           data['sid_frames'],
                                                           data['example_no']))

    fig.tight_layout()

    return ax
#end create_spinogram_plot()

def get_distance_traveled(x: np.ndarray, y: np.ndarray) -> float:
    """Estimate the distance traveled by a point in 2D space.
    Args:
        x (np.ndarray): x-coordinates of the points.
        y (np.ndarray): y-coordinates of the points.

    Returns:
        float: The total distance traveled.
    """
    return np.sum(np.sqrt(np.square(np.diff(x)) + np.square(np.diff(y))))
#end get_distance_traveled()

def get_midline_data(frame: np.ndarray) -> np.ndarray:
    """Estimate the hight of the mouse on the midline of the frame.

    Args:
        frame (np.ndarray): The frame data, expected to be a 2D array.

    Returns:
        np.ndarray: The estimated midline data, filtered and with values > 0.
    """
    mid = frame.shape[0] // 2
    result = np.median(frame[mid-5:mid+5, :], axis=0)
    result = signal.savgol_filter(result, 17, 3)
    #diag_plot(frame, result)
    return result[result > 0]
#end get_midline_data()

def diag_plot(frame, midline):
    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[5,1], height_ratios=[5,1])
    aximg = fig.add_subplot(gs[0, 0])
    axcbar = fig.add_subplot(gs[0, 1])
    axmid = fig.add_subplot(gs[1, 0], sharex=aximg)

    scm = aximg.imshow(frame)
    fig.colorbar(scm, cax=axcbar)
    axmid.plot(np.arange(midline.shape[0]), midline)
    plt.show(block=True)
#end diag_plot()
