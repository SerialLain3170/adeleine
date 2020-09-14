# pix2pixHD

## Summary
![net](https://github.com/SerialLain3170/Line-to-Color/blob/master/nohint_pix2pixHD/data/network.png)

- This model is the update version of pix2pix.
- The authors of this paper can generate high-resolution images by proposing coarse-to-fine generator and multi-scale discriminator.

## Usage
Execute the command line below
```
$ python train.py --data_path <DATA_PATH> --sketch_path <SKETCH_PATH>
```
`DATA_PATH`: path that contains image files  
`SKETCH_PATH`: path that contains line art files obtained by SKetchKeras  

## Result
- pix2pixHD nohint
![](../Data/nohint_comparison.png)

- pix2pixHD atari
![](https://github.com/SerialLain3170/Line-to-Color/blob/master/nohint_pix2pixHD/data/atari_result1.png)
![](https://github.com/SerialLain3170/Line-to-Color/blob/master/nohint_pix2pixHD/data/atari_result2.png)

- Batch size: 16
- Using Adam as optimizer
