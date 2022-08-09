# Car parts classify (Learning)

## 環境構築
```
cd parts-classify-learn
pip install -r requirements.txt
```

## datasetsフォルダの作成

```
datasets
├── label.txt            # クラス名を定義. 分類するクラスの名称を改行して書く. 
├── Images                # 画像データを保存するフォルダ. val用のデータはファイル名の先頭にvalと付ける. 
│   ├── hogehoge.jpg       # train用データ
│   ├── hogehoge2.jpg      # train用データ
│   ├── valhogehoge.jpg    # val用データ
│   └── valhogehoge2.jpg   # val用データ
├── BBoxes                # 正規化されたアノテーションデータ (label x y w h)
│   ├── hogehoge.txt       # ファイル名は画像データと同じファイル名で拡張子がtxt
│   ├── hogehoge2.txt
│   ├── valhogehoge.txt
│   └── valhogehoge2.txt
├── Labels                # 正規化されていないアノテーションデータ (xmin ymin xmax ymax label). BBoxesがある場合は不要.
│   ├── hogehoge.txt       # ファイル名は画像データと同じファイル名で拡張子がtxt
│   ├── hogehoge2.txt
│   ├── valhogehoge.txt
│   └── valhogehoge2.txt
├── Exclusion.txt          # 学習から除外する不適切画像を改行して書く. (ex. hogehoge.jpg)
└── Exclusion2.txt         # 学習から除外する不適切アノテーションを改行して書く. (ex. valhogehoge2.txt)　
    # Exclusion.txt, Exclusion2.txtのどちらかに格納されれば学習から除外される.
```

## 学習
 `config.yaml`を編集して学習の設定を行う

```bash
cd parts-classify-learn
python src/train.py
```

`/result/20******_**_YOLONano/`以下に  
`checkpoints`, `tensorbaord`, `code` と `log.train.txt` が生成される。  
  - `checkpoints` : モデルのweight  
  - `tensorboard` : tensorbaordのlog  
  - `code` : 実行した時のコードやconfig  
  - `log.train.txt` : trainのlog  

## 評価 
```
cd parts-classify-learn
python val.py ../results/20******_**_YOLONano
```
`result/20******_**_YOLONano/`に`log.val.txt`

## 結果
[best modelを使用したAPI]()

---
## 引用 
[YOLO Nano: a Highly Compact You Only Look Once
Convolutional Neural Network for Object Detection](https://arxiv.org/pdf/1910.01271.pdf) implemented in PyTorch.
