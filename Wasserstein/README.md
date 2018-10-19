# pix2pix + wasserstein metrics

## Summary
![net](https://github.com/SerialLain3170/Line-to-Color/blob/master/Wasserstein/net.png)
- GeneratorはUNetベース。中間層にはResidual Blocksを用いている。
- UpsamplingにはNearest Neighbor Upsampling -> Convolutionを導入。

## Usage
`line_path`に線画を格納、`color_path`に着色画像を格納し以下を実行。
```py
$ python train.py
```

## Result
私の環境で生成した画像を以下に示す。
![image](https://github.com/SerialLain3170/Line-to-Color/blob/master/Wasserstein/result.png)

- 損失関数としては上記画像のようにzero-centered gradient penalties,one-centered gradient penaltiesとadversarial lossのみを考慮しているが見栄え的にソースコード上はzero-centered gradient penaltiesを用いた。
- バッチサイズは3
- gradient penaltyの重みは10.0、content lossの重みも10.0
- 最適化手法はAdam(α=0.0001, β1=0.5, β2=0.9)
