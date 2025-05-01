


import argparse
import os

from moseq_spinogram.spinogram import plot_clustered_corpus, plot_many_single_syllable, plot_one_single_syllable, plot_syllable_corpus
from moseq_spinogram.util import get_syllable_id_mapping


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    pone_parser = subparsers.add_parser('plot-one', help="Plot just one example of a given syllable")
    pone_parser.add_argument('syllable_id', type=int)
    pone_parser.set_defaults(func=plot_one_single_syllable)

    pmny_parser = subparsers.add_parser('plot-many', help="Plot many examples of a given syllable")
    pmny_parser.add_argument('syllable_id', type=int)
    pmny_parser.add_argument('--max-examples', default=100, type=int, help="Max number of syllable examples to plot")
    pmny_parser.set_defaults(func=plot_many_single_syllable)

    pcorp_parser = subparsers.add_parser('plot-corpus', help="Plot one example of each syllable")
    pcorp_parser.add_argument('--max-syllable', default=100, type=int, help="Max number of syllables to plot")
    pcorp_parser.add_argument('--separate', action="store_true", help="Output seperate plots for each syllable.")
    pcorp_parser.add_argument('--max-examples', default=10, type=int, help="Max number of examples to plot")
    pcorp_parser.set_defaults(func=plot_syllable_corpus)

    pclust_parser = subparsers.add_parser('plot-clustered', help="Plot one example of each syllable, clustered by behavioral distance")
    pclust_parser.add_argument('--max-syllable', default=100, type=int, help="Max number of syllables to plot")
    dist_choices = ['ar[init]', 'ar[dtw]', 'scalars', 'pca[dtw]']
    pclust_parser.add_argument('--distance', choices=dist_choices, default=dist_choices[0], help="Metric used to compute behavioral distance.")
    clust_choices = ['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward']
    pclust_parser.add_argument('--linkage', choices=clust_choices, default=clust_choices[0], help="Method used to compute linkage (see scipy.cluster.hierarchy.linkage for more info.).")
    pclust_parser.add_argument('--show-pca', action='store_true', help="Show PCA value plots")
    pclust_parser.set_defaults(func=plot_clustered_corpus)

    for _, subp in subparsers.choices.items():
        subp.add_argument('index')
        subp.add_argument('model')
        subp.add_argument('--sort', action='store_true', help="Sort syllables")
        subp.add_argument('--count', choices=['usage', 'frames'], default='usage', help="metric for sorting")
        subp.add_argument('--num-samples', default=10, help="Number of evenly spaced samples to take over syllable duration for plotting.")
        subp.add_argument('--name', default='spinogram', help="output basename.")
        subp.add_argument('--dir', help="Output directory")
        subp.add_argument('--color', default="red", help="Color to use for the spinogram plot.")
        subp.add_argument('--save-data', action='store_true', help="Save the spinogram data to JSON.")
        subp.add_argument('--no-plot', action='store_true', help="Do not output plot(s).")

        pick_choices = ['median', 'longest', 'shortest', 'shuffle']
        subp.add_argument('--pick', choices=pick_choices, default=pick_choices[0], help="Method for choosing which syllable instance(s) to plot.")

    args = parser.parse_args()

    if args.dir is None:
        args.dir = os.path.dirname(os.path.abspath(args.index))
    os.makedirs(args.dir, exist_ok=True)

    args.name = os.path.join(args.dir, args.name)

    label_map = get_syllable_id_mapping(args.model)
    if args.sort and args.count == 'usage':
        args.label_map = { itm['usage']: itm for itm in label_map }

    elif args.sort and args.count == 'frames':
        args.label_map = { itm['frames']: itm for itm in label_map }
    
    else:
        args.label_map = { itm['raw']: itm for itm in label_map }


    args.func(args)



if __name__ == "__main__":
    main()