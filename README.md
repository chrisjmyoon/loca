# LoCA: Longitudinal Component Analysis
This project builds on Longitudinal Self-supervised Learning (LSSL) which aligns longitudinal changes (aging effects) along a dimension in the latent space. LoCA generalizes LSSL to algining longitudinal changes along more than a single dimension, which allows for representing different longitudinal changes for different groups (normal aging vs disease aging). 

## Method
**Note: this method was developed when the encoder was using a tanh activation which made training more unstable. The loss terms may have simply made training the encoder more stable instead of meaningfully regularizing the representations**

For a pair of longitudinal scans $X_1, X_2$, LoCA aligns $\delta_z = z_2 - z_1$ along a set of $K$ longitudinal dimensions of $tau$. 

The residual loss aligns $\delta_z$ along $K$ $tau$ vectors:

$$\mathcal{L}_{residual} = minimize( \delta_z - \sum_{k=1}^{K}proj_{tau_k} \delta_z)$$

To prevent $\delta_z$ from degenerating to $0$, we also implement a normalized variant:

$$\mathcal{L}_{residual} = minimize(\frac{\delta_z}{\|\delta_z\|} - \sum_{k=1}^{K}proj_{tau_k} \frac{\delta_z}{\|\delta_z\|})$$

The orthogonality loss encourages the learned $tau$ vectors to be orthogonal

$$\mathcal{L}_{orthogonality} = \sum_{k=1}^{K-1}\sum_{m=k+1}^{K} cos(tau_k, tau_m)$$

The correlation loss encourages information across $tau$ vectors to be uncorrelated

$$ \mathcal{L}_{correlation} = \sum_{k=1}^{K-1}\sum_{m=k+1}^{K} \dfrac{cov(\alpha_k, \alpha_m)}{\sigma_{\alpha_k}\sigma_{\alpha_m}}$$
where $\alpha_i$ is the scalar component of $\delta_z$ along $tau_i$

## Setup
Install dependencies using the conda environment.yml file
```
conda env create -f environment.yml
```
Developed using **Python 3.8.5** and **PyTorch 1.7.0**

## Running experiments
Visdom is used to track the learning process. Start visdom before running the main script.
The main script is called by:
```
python main.py configs/experiments/$config_name.hocon --exp $exp_name
```
For example, to call LoCA:
```
python main.py configs/experiments/loca.hocon --exp loca
```
## Code organization
`main.py`initializes an experiment based on the command arguments for the configuaration file and experiment type.

#### configs/
Contains `.hocon` configuration files specifiying configurations such as hyperparameters, data path, save path, etc. Config files are hierarchical. 
`base.hocon` contains configurations common to all experiments, `adni.hocon` and `aud.hocon` contains configurations specific to each dataset, `experiment/$exp_name.hocon` contains configurations specific to each experiment.

#### experiments/
`Experiment` initializes the dataloaders and an agent responsible for most of the training/model logic. `Experiment` has multiple endpoints such as `train`, `visualize`, `eval_dcorr` which can be specified in the experiment config file. `Experiment` also sets up the cross validation folds based on the the `rand_seed` parameter in the configuration files.

#### agents/
`Agent` contains the training logic for different models. `agent.ml_logic()` returns `loss_dict` which is a dictionary of all loss terms. The loss items in `loss_dict` are visualized through the plotters.

#### graphs/
Contains model architectures and loss functions for baselines such as `SimCLR` and `VAE`.

#### datasets/
`PairedDataset` returns longitudinal pairs of scans, while `SingleDataset`returns individual scans and associated meta data (age, dx).
Both datasets initializes the scans to memory as a singleton.

#### splits/
Splits contains filenames of paired longitudinal scans in (img1.txt, img2.txt) for ADNI and (aud_img1.txt, aud_img2.txt) for AUD. 
Pairs are split by classes by `split_pairs_by_class.py` and bad filepaths without a corresponding scan are removed by `clean.py`. `combine_split.py` combines multiple classes into a single class, for example as a single disease class. 
The `Experiment` class separates the pairs to cross validation splits saved in `splits/cv/`.

#### utils/
Contains util helpers: 
* `configs.py` for loading the `.hocon` as an object 
* `fs.py` for saving and loading files
* `meters.py` for meter objects to keep track of metrics
* `viz.py` for visdom plotters

#### results/
Results of evaluations are saved here.

#### artifacts/
Models are saved here, organized by `rand_seed` and `exp_name`.