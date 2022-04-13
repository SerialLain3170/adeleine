# Adeleine

## Caution!

- You can upload only `.png` files.
- The point-based model can accept images that have limited shapes, which are two to the Nth power (ex: 256, 512, 1024, etc).

## Summary
![](./data/result1.png)
![](./data/result2.png)
![](./data/result3.png)

- A simple GUI application of colorization based on three types of hints!
  - Reference: colorization based on reference images
  - Flatten: colorization based on scribble hints
  - Point: colorization based on point hints

## Getting Started

### 0. Download pre-trained file
- Download `point_model.pt` from [the link](https://github.com/SerialLain3170/Colorization/releases/tag/v0.1.0-alpha) and move the file to `Adeleine/ckpts/`

### 1. Start Adeleine
- Start application via the command below

```
$ python server.py --point ckpts/point_model.pt
```

### (Optional)
- You can try pretrained files for reference or flatten

```
$ python server.py --ref <REF_PRETRAIN_PATH> --flat <FLAT_PRETRAIN_PATH>
```
