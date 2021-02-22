# White Box Colorization

## Summary
![](./data/concept.png)

- This implementation is based on [SPADE](https://arxiv.org/pdf/1903.07291.pdf)

## Usage

### Training Phase
Execute the command line below.

```bash
$ python train.py --data_path <DATA_PATH> --sketch_path <SKETCH_PATH> --ss_path <SS_PATH>
```
- `DATA_PATH`: The name of the directory that contains color images
- `SKETCH_PATH`: The name of the directory that contains line arts obtained by SketchKeras
- `SS_PATH`: The name of the directory that contains quantized color images

File names of `DATA_PATH` must correspond to those of `SKETCH_PATH`. The examples of dataset structures are as follows.

```
ex1

DATA_PATH - file1.jpg
          - file2.jpg
          ...

SKETCH_PATH - file1.jpg
            - file2.jpg
            ...

SS_PATH - file1.jpg
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

SS_PATH - file1.jpg
        - file2.jpg
        - file3.jpg
        - file4.jpg
        ...
```

## Results
![](./data/result1.png)
![](./data/result2.png)

- I think that colorization networks containing SPADE tend to fade colors...
