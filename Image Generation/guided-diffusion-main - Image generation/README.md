# improved-diffusion

The code is adjusted on the basis of [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672).


## Installation

Clone this repository and navigate to it in your terminal. Then run:

```
pip install -e .
```

This should install the `improved_diffusion` python package that the scripts depend on.


## Training

To train your model, you should first decide some hyperparameters. We will split up our hyperparameters into three groups: model architecture, diffusion process, and training flags. Here are some reasonable defaults for a baseline:

```
MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--lr 3e-4 --batch_size 64"
```

you can run an experiment like so:

```
python scripts/image_train.py --data_dir path/to/images $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```

## Sampling

Once you have a path to your model, you can generate a large batch of samples like so:

```
python scripts/image_sample.py --model_path /path/to/model.pt $MODEL_FLAGS $DIFFUSION_FLAGS
```



```bash
MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--lr 3e-4 --batch_size 64"
```

Class-conditional ImageNet-64 model (270M parameters, trained for 250K iterations) [[checkpoint](https://openaipublic.blob.core.windows.net/diffusion/march-2021/imagenet64_cond_270M_250K.pt)]:


