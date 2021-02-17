# Video colorization with a few references

## TODO
- [ ] Pushing data preparation code 
- [ ] Adding the description of `separate.txt` in dataset.py

## Summary
![](./data/concept.png)

- This implementation is based on [this paper](https://arxiv.org/pdf/2003.10685.pdf)
- Given the first frame and the last frame of one scene, this model colorizes the rest.

## Usage

### Training Color Transform Network Phase
Execute the command line below.

```bash
$ python train.py --data_path <DATA_PATH> --sketch_path <SKETCH_PATH> --dist_path <DIST_PATH>
```
- Descriptions
    - `DATA_PATH`: The name of the directory that contains color images
    - `SKETCH_PATH`: The name of the direcotry that contains line arts obtained by SketchKeras
    - `DIST_PATH`: The name of the direcotry that contains distance field images of lines arts obtained by SketchKeras

- Directory structures
    - `DATA_PATH`, `SKETCH_PATH` and `DIST_PATH` must have the same names of directories and files like the example below.
    - File names without suffix must be in numerical order (ex: 0, 1, 2, 3...).

```
example

DATA_PATH - anime_dir1 - 0.jpg
                       - 1.jpg
                       ...
          - anime_dir2 - 0.jpg
                       - 1.jpg
                       ...
          ...

SKETCH_PATH - anime_dir1 - 0.jpg
                         - 1.jpg
                         ...
            - anime_dir2 - 0.jpg
                         - 1.jpg
                         ...
            ...

DIST_PATH - anime_dir1 - 0.jpg
                       - 1.jpg
                       ...
          - anime_dir2 - 0.jpg
                       - 1.jpg
                       ...
          ...
```

### Training Temporal Constraint Network Phase
Execute the command line below.

```bash
$ python refine.py --data_path <DATA_PATH> --sketch_path <SKETCH_PATH> --dist_path <DIST_PATH> --pre_path <PRETRAIN_PATH>
```
- Descriptions
    - `DATA_PATH`: The name of the directory that contains color images
    - `SKETCH_PATH`: The name of the direcotry that contains line arts obtained by SketchKeras
    - `DIST_PATH`: The name of the direcotry that contains distance field images of lines arts obtained by SketchKeras
    - `PRETRAIN_PATH`: The name of pretrained Color Transform Network file.

- Directory structures
    - `DATA_PATH`, `SKETCH_PATH` and `DIST_PATH` must have the same names of directories and files like the example below.
    - File names without suffix must be in numerical order (ex: 0, 1, 2, 3...).

## Result

### Yakusoku no Neverland
![](./data/never_color1.gif)
![](./data/never_color2.gif)
![](./data/never_color3.gif)

### Sakura Taisen
![](./data/sakura1_color1.gif)
![](./data/sakura1_color2.gif)
![](./data/sakura1_color3.gif)


### Magic Knight Rayearth
![](./data/rayearth1_color1.gif)
![](./data/rayearth1_color2.gif)
![](./data/rayearth1_color3.gif)
