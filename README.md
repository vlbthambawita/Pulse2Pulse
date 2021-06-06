# Pulse2Pulse - DeepFake ECG Generator (Development-repository) 

# [Paper](https://doi.org/10.1101/2021.04.27.21256189) |  Pre-generated Data - ([OSF](https://osf.io/6hved/), [Kaggle](https://www.kaggle.com/vlbthambawita/deepfake-ecg))

## Pre-trained DeepFake ECG generator and it's funtionaltities can be found here: - [GitHub (DeepFake ECG)](https://github.com/vlbthambawita/deepfake-ecg)

In this repository, we present how to train the Pulse2Pulse generative adversarial network which can generate DeepFake ECGs as discussed in our original paper. We maintain a seperate GitHub repository for generating random DeepFake ECGs from our pretrained model. 

## Trainning Pulse2Pulse GAN to generate DeepFake ECGs. 


```python
# To train (check the pulse2pulse_train.py file for more information)
python pulse2pulse_train.py train \ # Three options: train, retrain, inference, check
    --exp_name "test_exp_1" \ # A name to the experiment
    --data_dirs ./data_dir_1 ./data_dir_2 \ # data directories (check sample_ecg_data directory for the format)
    --checkpoint_interval 25 \
    --num_epochs 4000 \
    --start_epoch 0 \
    --bs 32 \
    --lr 0.0001 \
    --b1 0.5 \
    --b2 0.9 \
```
```python
# To retrain the above experiment from a checkpoint at epoch 100
python pulse2pulse_train.py retrain --exp_name "test_exp_1" \
    --num_epochs 3000 \
    --start_epoch 100 \ #start epoch number is 100 assuming that the checkpoint used to restart the trianing is 100
    --checkpoint_path ".\checkpoint\chck_100.pt"
```
More parameters are in the pulse2pulse_train.py file. For example, output directory, tensorboard directory etc. can be changed in this file. 

---
## Citation:
```latex
@article {Thambawita2021.04.27.21256189,
	author = {Thambawita, Vajira and Isaksen, Jonas L. and Hicks, Steven A. and Ghouse, Jonas and Ahlberg, Gustav and Linneberg, Allan and Grarup, Niels and Ellervik, Christina and Olesen, Morten Salling and Hansen, Torben and Graff, Claus and Holstein-Rathlou, Niels-Henrik and Str{\"u}mke, Inga and Hammer, Hugo L. and Maleckar, Molly and Halvorsen, P{\r a}l and Riegler, Michael A. and Kanters, J{\o}rgen K.},
	title = {DeepFake electrocardiograms: the key for open science for artificial intelligence in medicine},
	elocation-id = {2021.04.27.21256189},
	year = {2021},
	doi = {10.1101/2021.04.27.21256189},
	publisher = {Cold Spring Harbor Laboratory Press},
	URL = {https://www.medrxiv.org/content/early/2021/05/10/2021.04.27.21256189.1},
	eprint = {https://www.medrxiv.org/content/early/2021/05/10/2021.04.27.21256189.1.full.pdf},
	journal = {medRxiv}
}

```


## Contact us:
vajira@simula.no | michael@simula.no