# imagenet-grad-noise

This code was adapted from https://github.com/jiweibo/ImageNet


## PyTorch

- Example usage:
```

python train.py  PATHTOIMAGENET -save_noise -save_sharpness --save-dir PATH_TO_SAVE_LOGS -j 16 -ls constant --lr 0.001 -noise_size 600 -sharpness_batches 600 -b 250 -sf 7000

```
The above describes one particular plot for the constant step size training.
The stats needs to be smoothed to avoid periodicity in text. 

## Brief descriptions


Sharpness is computed using power method from https://github.com/leiwu0/sgd.stability


- save-dir - str, name of exp
- save_noise - bool, whether to compute and store noise/ grad norm
- noise_size - int, number of batch used to compute the full grad (sample number = noise_size x batch_size)
- save_sharpness - bool, whether to compute and store sharpness
- sharpness_batches - int, number of batch used to compute the sharpness (sample number = noise_size x batch_size)
- epoch_interval - int, number of train epcohs per computation of noise and sharpness


