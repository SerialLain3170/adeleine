# UserHint

## Summary
![here](https://github.com/SerialLain3170/Line-to-Color/blob/master/UserHint/network.png)

- 線画と一緒にカラーヒントを与えて、UNet + Discriminatorで生成
- 線画だけでなく、VGG16を通した線画特徴量も中間層に与える

## Results
私の環境で生成した例を以下に示す。
![here](https://github.com/SerialLain3170/Line-to-Color/blob/master/UserHint/example.png)

- バッチサイズは16
- 最適化手法はAdam(α=0.0002, β1=0.5)
- 損失関数としてはAdversarial lossとContent loss。重みはそれぞれ0.001と1.0
