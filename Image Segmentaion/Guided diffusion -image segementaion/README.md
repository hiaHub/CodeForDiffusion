The implementation of Denoising Diffusion Probabilistic Models presented in the paper is based on [openai/guided-diffusion](https://github.com/openai/guided-diffusion).
## Requirement

``pip install -r requirement.txt``
## Data

We used the classic dataset MSD、BRATS2021、ROSE in medicine, which you can download from the link below
- http://medicaldecathlon.com/dataaws/
- https://www.synapse.org/#!Synapse:syn25829067/wiki/610863
- https://imed.nimte.ac.cn/dataofrose.html


## Usage  
 ``python scripts/segmentation_train.py --data_name ISIC --data_dir input data direction --out_dir output data direction --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 1e-4 --batch_size 8``
    
 ``python scripts/segmentation_sample.py --data_name ISIC --data_dir input data direction --out_dir output data direction --model_path saved model --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --num_ensemble 5``

 ``python scripts/segmentation_env.py --inp_pth *folder you save prediction images* --out_pth *folder you save ground truth images*``





