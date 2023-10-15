The implementation of Denoising Diffusion Probabilistic Models presented in the paper is based on [openai/guided-diffusion](https://github.com/openai/guided-diffusion).
## Requirement

``pip install -r requirement.txt``
## Data

We used the classic dataset HKU-SZH X-ray Set、PALM、CAT in medicine, which you can download from the link below
- https://paperswithcode.com/dataset/shenzhen-hospital-x-ray-set
- https://aistudio.baidu.com/competition/detail/87/0/introduction
- https://aistudio.baidu.com/aistudio/datasetdetail/106986


## Usage
```bash
python test.py --conf_path confs/face_example.yml
```
Find the output in `./log/face_example/inpainted`


