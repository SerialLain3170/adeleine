# pix2pixHD

## Summary
![net](https://github.com/SerialLain3170/Line-to-Color/blob/master/pix2pixHD/network.png)
- Coarse-to-fine GeneratorとMulti-scale discriminatorによって多段的に高解像度へ対応
- Instance-level Feature Embeddingでインスタンス毎に違うマッピングを行う

## Usage
```py
$ python pretrain.py
```
でGlobal Generatorを事前に学習、そして
```py
$ python train.py
```
でLocal Enhancerも含めて全て学習

## Result
私の環境で生成した例を以下に示す。
![result](https://github.com/SerialLain3170/Line-to-Color/blob/master/pix2pixHD/visualize_125.png)

- バッチサイズは4
- 最適化手法はAdam(α=0.0002, β1=0.5)
- Multi-Scale Discriminatorによるlossの重みは10
- まだ学習はしっかり出来ていない、というかGlobal Generatorの事前学習どうやってるんだろうか.....
