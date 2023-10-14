The implementation of Denoising Diffusion Probabilistic Models presented in the paper is based on [openai/guided-diffusion](https://github.com/openai/guided-diffusion).


## Data

We used the classic dataset HKU-SZH X-ray Set、PALM、CAT in medicine, which you can download from the link below
- https://paperswithcode.com/dataset/shenzhen-hospital-x-ray-set
- https://aistudio.baidu.com/competition/detail/87/0/introduction
- https://aistudio.baidu.com/aistudio/datasetdetail/106986


## Usage

We set the flags as follows:
```
MODEL_FLAGS="--image_size 256 --num_channels 128 --class_cond True --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
TRAIN_FLAGS="--lr 1e-4 --batch_size 10"
CLASSIFIER_FLAGS="--image_size 256 --classifier_attention_resolutions 32,16,8 --classifier_depth 4 --classifier_width 32 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True"
SAMPLE_FLAGS="--batch_size 1 --num_samples 1 --timestep_respacing ddim1000 --use_ddim True"
```
To train the classification model, run
```
python scripts/classifier_train.py --data_dir path_to_traindata --dataset brats_or_chexpert $TRAIN_FLAGS $CLASSIFIER_FLAGS
```
To train the diffusion model, run
```
python scripts/image_train.py --data_dir --data_dir path_to_traindata --datasaet brats_or_chexpert  $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```
The model will be saved in the *results* folder.

For image-to-image translation to a healthy subject on the test set, run
```
python scripts/classifier_sample_known.py  --data_dir path_to_testdata  --model_path ./results/model.pt --classifier_path ./results/classifier.pt --dataset brats_or_chexpert --classifier_scale 100 --noise_level 500 $MODEL_FLAGS $DIFFUSION_FLAGS $CLASSIFIER_FLAGS  $SAMPLE_FLAGS 
```




