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
