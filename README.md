# Moseq-Spinogram
Package to generate Spinograms from moseq data.

## Install
`moseq-spinogram` can be installed into a `moseq2-app` environment, or you could make your own virtual environment:
```sh
conda create -n moseq-spinogram python=3.7
conda activate moseq-spinogram
```

Please ensure the following dependencies are installed:
```sh
pip install git+https://github.com/dattalab/moseq2-viz.git
```

Then install this package:
```sh
pip install git+https://github.com/tischfieldlab/moseq-spinogram.git
```


## Usage

```bash
spinogram --help
```

There are a few sub commands depending on the extent you want to generate spinograms. These all mostly share the same options.
```
plot-one            Plot just one example of a given syllable
plot-many           Plot many examples of a given syllable
plot-corpus         Plot one example of each syllable
plot-clustered      Plot one example of each syllable, clustered by behavioral distance
```