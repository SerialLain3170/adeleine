# pix2pixHD

## Summary
![net](data/network.png)

## Usage
Execute the command line below
```
$ python train.py --data_path <DATA_PATH> --sketch_path <SKETCH_PATH>
```
`DATA_PATH`: The name of the directory that contains image files  
`SKETCH_PATH`: The name of the directory that contains line art files obtained by SKetchKeras  

## Result

| Methods | Results |
| ---- | ---- |
| pix2pixHD nohint | ![](../Data/nohint_comparison.png) |
| pix2pixHD atari | ![](data/atari_result1.png)![](data/atari_result2.png) |

- Batch size: 16
- Using Adam as optimizer
