# Pulse2Pulse - DeepFake ECG Generator (Development-repository) 

# [Paper](https://doi.org/10.1101/2021.04.27.21256189) |  Pre-generated Data - ([OSF](https://osf.io/6hved/), [Kaggle](https://www.kaggle.com/vlbthambawita/deepfake-ecg))

## Pre-trained DeepFake ECG generator and it's funtionaltities can be found here: - [GitHub (DeepFake ECG)](https://github.com/vlbthambawita/deepfake-ecg)

In this repository, we present how to train the Pulse2Pulse generative adversarial network which can generate DeepFake ECGs as discussed in our original paper. We maintain a seperate GitHub repository for generating random DeepFake ECGs from our pretrained model. 

## Training Pulse2Pulse GAN to generate DeepFake ECGs

All parameters are configured via a YAML file. A full example is provided in `config.yaml`.

### Usage

```bash
python pulse2pulse_train.py --config config.yaml
```

### Config file reference

```yaml
# ============================================
# Pulse2Pulse Training Configuration
# ============================================

# Action: train | retrain | inference | check
action: train

# Hardware
device_id: 0                        # CUDA device index

# Experiment
exp_name: pulse2pulse_exp_ptbxl_full  # Name used for checkpoints and W&B run
out_dir: /path/to/output              # Directory where checkpoints are saved

# Weights & Biases
wandb_project: Pulse2Pulse            # W&B project name

# ============================================
# Dataset
# ============================================
dataset: ptbxl          # simple | ptbxl
                        # simple: custom .asc files
                        # ptbxl:  PTB-XL dataset (folds 1-8 train, 9 val, 10 test)

# PTB-XL options (used when dataset: ptbxl)
ptbxl_path: /path/to/ptb-xl/1.0.3  # Root directory containing ptbxl_database.csv
ptbxl_sampling_rate: 500            # 100 or 500 Hz
                                    # Leads used: I, II, V1, V2, V3, V4, V5, V6

# Simple dataset options (used when dataset: simple)
data_dirs:
  - /path/to/ecg_data_dir           # One or more directories of .asc files

# Limit samples per split for quick testing (set to null to use all data)
max_samples: null                   # e.g. 100 for a quick smoke-test run

# ============================================
# Hyper parameters
# ============================================
bs: 32                  # Batch size
lr: 0.0001              # Learning rate
b1: 0.5                 # Adam beta1
b2: 0.9                 # Adam beta2
num_epochs: 4000        # Total training epochs
start_epoch: 0          # Starting epoch (set automatically on retrain)
ngpus: 1                # Number of GPUs
checkpoint_interval: 25 # Save checkpoint every N epochs
model_size: 50          # WaveGAN model size parameter
lmbda: 10.0             # Gradient penalty regularization factor (WGAN-GP)

# Checkpoint path (required for action: retrain)
checkpoint_path: ""
```

### Quick test run

To verify the pipeline with a small subset of data before full training:

```yaml
max_samples: 100
num_epochs: 5
```

### Retrain from a checkpoint

Set `action: retrain` and provide the checkpoint path:

```yaml
action: retrain
checkpoint_path: /path/to/output/my_exp/checkpoints/my_exp_epoch:100.pt
```

---
## Citation:
```latex

@article{cite-key,
	author = {Thambawita, Vajira and Isaksen, Jonas L. and Hicks, Steven A. and Ghouse, Jonas and Ahlberg, Gustav and Linneberg, Allan and Grarup, Niels and Ellervik, Christina and Olesen, Morten Salling and Hansen, Torben and Graff, Claus and Holstein-Rathlou, Niels-Henrik and Str{\"u}mke, Inga and Hammer, Hugo L. and Maleckar, Mary M. and Halvorsen, P{\aa}l and Riegler, Michael A. and Kanters, J{\o}rgen K.},
	da = {2021/11/09},
	date-added = {2021-11-28 10:10:31 +0530},
	date-modified = {2021-11-28 10:10:31 +0530},
	doi = {10.1038/s41598-021-01295-2},
	id = {Thambawita2021},
	isbn = {2045-2322},
	journal = {Scientific Reports},
	number = {1},
	pages = {21896},
	title = {DeepFake electrocardiograms using generative adversarial networks are the beginning of the end for privacy issues in medicine},
	ty = {JOUR},
	url = {https://doi.org/10.1038/s41598-021-01295-2},
	volume = {11},
	year = {2021},
	Bdsk-Url-1 = {https://doi.org/10.1038/s41598-021-01295-2}
}


```


## Contact us:
vajira@simula.no | michael@simula.no
