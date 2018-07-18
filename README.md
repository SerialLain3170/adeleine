# pix2pix  
## Summary
論文は[こちら](https://arxiv.org/abs/1611.07004 "here")  
- ネットワーク構造としてはUNetを用いている
- 損失関数に平均絶対値誤差に加え、real/fakeを判別するAdversarial lossを考慮

## Dataset
safebooruからイラストを10000枚集め、うまい具合に128 * 128にした後64 * 64をクラップ、それらをopencvを用いて線画抽出した  
入力データを線画、教師データを着色画像とした

## Usage
線画、着色画像をそれぞれ`.npy`ファイルとして格納し、以下のコマンドを実行
```bash
$ python colorization.py
```

## Result
私の環境で着色した画像を以下に示す。
![result](https://github.com/SerialLain3170/Line-to-Color/blob/master/result.png)
![result2](https://github.com/SerialLain3170/Line-to-Color/blob/master/result2.png)
- バッチサイズは10
- 最適化手法はAdam(α=0.0002, β1=0.5)
- 平均絶対値誤差の重みは10.0
