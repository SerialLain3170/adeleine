# Augmented-Self Reference and Dense Semantic Correspondence

## Summary
![](./data/concept.png)

- This directory implements colorization with reference images based on [this paper](https://arxiv.org/pdf/2005.05207.pdf)


## Usage

### Training Phase
Execute the command line below.

```bash
$ python train.py --data_path <DATA_PATH> --sketch_path <SKETCH_PATH>
```
- `DATA_PATH`: The name of the directory that contains color images
- `SKETCH_PATH`: The name of the directory that contains line arts obtained by SketchKeras

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

| Results |
| ---- |
| ![](./data/result1.png) |
| ![](./data/result2.png) |
| ![](./data/result3.png) |

- `AdaIN` corresponds to my implementation in `../reference_adain` and `SCFT` corresponds to this implementation.
- While `AdaIN` sprays color on line arts, `SCFT` is able to colorize by recognizing correspondences between reference images and line arts

### 2021/04/20 added
- More perturbation leads to dealing with more dynamic reference images

| Results |
| ---- |
| ![](./data/result4.png) |
| ![](./data/result5.png) |
