
### Data Prepare

We used the classic dataset BRATS2020 in medicine, which you can download from the link below
- https://www.med.upenn.edu/cbica/brats2020/


### Training/Resume Training
1. Download the checkpoints from given links.
2. Set `resume_state` of configure file to the directory of previous checkpoint. 

then run:
```python
netG_label = self.netG.__class__.__name__
self.load_network(network=self.netG, network_label=netG_label, strict=False)
```

3. Run the script:

```python
python run.py -p train 
```


### Test

```python
python run.py -p test
```

### Evaluation

1. Run the script:

```python
python eval.py -s [ground image path] -d [sample image path]
```
