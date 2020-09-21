# Colorization With Reference Images By Using AdaIN

## Summary
![](https://github.com/SerialLain3170/Colorization/blob/master/reference_adain/data/generator.png)

- This directory implements colorization with reference images by using Adaptive Instance Normalization(AdaIN).

## Usage

### Traning Phase
Execute the command line below.

```bash
$ python train.py --data_path <DATA_PATH> --sketch_path <SKETCH_PATH>
```
- `DATA_PATH`: The name of the directory that contains color images
- `SKETCH_PATH`: The name of the direcotyr that contains line arts obtained by SketchKeras

File names of `DATA_PATH` must correspond to those of `SKETCH_PATH`. The examples of dataset structures are as follows.

```
ex1

DATA_PATH - file1.jpg
          - file2.jpg
          ...

SKETCH_PATH - file1.jpg
            - file2.jpg
            ...
```

```
ex2

DATA_PATH - dir1 - file1.jpg
                 - file2.jpg
          - dir2 - file3.jpg
                 - file4.jpg
          ...
          
SKETCH_PATH - file1.jpg
            - file2.jpg
            - file3.jpg
            - file4.jpg
            ...
```

## Result
![](https://github.com/SerialLain3170/Colorization/blob/master/reference_adain/data/res1.png)
![](https://github.com/SerialLain3170/Colorization/blob/master/reference_adain/data/res2.png)
![](https://github.com/SerialLain3170/Colorization/blob/master/reference_adain/data/res3.png)
