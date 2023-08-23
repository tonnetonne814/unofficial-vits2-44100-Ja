# unofficial vits2-TTS implementation in pytorch (44100Hz 日本語版)
**VITS2: Improving Quality and Efficiency of Single-Stage Text-to-Speech with Adversarial Learning and Architecture Design**

このリポジトリは、 44100Hzの日本語音声を学習および出力できるように編集した、[unofficial vits2-TTS implementation in pytorch](https://github.com/p0p4k/vits2_pytorch)です。2023/08/22 update 5までが反映されています。

<img src="./resources/image.png" width="100%">

## 1. 環境構築

Anacondaによる実行環境構築を想定する。

0. Anacondaで"vits2"という名前の仮想環境を作成する。[y]or nを聞かれたら[y]を入力する。
    ```sh
    conda create -n vits2 python=3.8    
    ```
0. 仮想環境を有効化する。
    ```sh
    conda activate vits2 
    ```
0. このレポジトリをクローンする（もしくはDownload Zipでダウンロードする）
    
    ```sh
    git clone https://github.com/tonnetonne814/unofficial-vits2-44100-Ja.git 
    cd unofficial-vits2-44100-Ja # フォルダへ移動
    ```

0. [https://pytorch.org/](https://pytorch.org/)のURLよりPyTorch1.13.1をインストールする。
    
    ```sh
    # OS=Linux, CUDA=11.7 の例
    pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
    ```

0. その他、必要なパッケージをインストールする。
    ```sh
    pip install -r requirements.txt 
    ```
0. Monotonoic Alignment Searchをビルドする。
    ```sh
    cd monotonic_align
    mkdir monotonic_align
    python setup.py build_ext --inplace
    cd ..
    ```

## 2. データセットの準備

[JSUT Speech dataset](https://sites.google.com/site/shinnosuketakamichi/publication/jsut)によるBasic5000音源、[ITAコーパス](https://github.com/mmorise/ita-corpus)によるEmotion音源とRecitation音源、及び自作データセット音源による、44100Hzでの学習を想定する。

-  JSUT Basic5000
    1. [JSUT Speech dataset](https://sites.google.com/site/shinnosuketakamichi/publication/jsut)をダウンロード及び展開する。
    1. 展開したフォルダの中にあるbasic5000フォルダを指定して、以下を実行する。
        ```sh
        python3 ./dataset/preprocess.py --dataset_name jsut --folder_path ./path/to/jsut_ver1.1/basic5000/ --sampling_rate 44100
        ```
-  [ITAコーパス](https://github.com/mmorise/ita-corpus) (例：[あみたろの声素材工房](https://amitaro.net/) 様)
    1. [ITAコーパス読み上げ音声](https://amitaro.net/voice/corpus-list/ita/)をダウンロードし、展開する。
    1. RECITATION音源が格納されているrecitationフォルダと、EMOTION音源が格納されているemotionフォルダを準備し、2つのフォルダが格納されているフォルダを指定して、以下を実行する。
        ```sh
        python3 ./dataset/preprocess.py --dataset_name ita --folder_path ./path/to/ita_corpus/ --sampling_rate 44100
        ```
        > ⚠音源は、ファイル名の001や002等の3桁の数字で区別するので、3桁の数字を含むこと。

        > ⚠音源を格納している2つのフォルダ名は、それぞれ”recitation”と"emotion"にすること。

-   自作データセット(単一話者)
    1. 以下の要素に注意して、読み上げ音声を準備する。([What makes a good TTS dataset](https://github.com/coqui-ai/TTS/wiki/What-makes-a-good-TTS-dataset)より)
        - テキストや発話の長さが正規分布感になってること。
        - テキストデータと発話音声に間違いがないこと。
        - 背景ノイズが無いこと。
        - 発話音声データ間で、話し方が似通っていること。
        - 使用する言語の音素を網羅していること。
        - 声色や音程の違い等をできるだけ自然に録音していること。
    1. `./dataset/homebrew/transcript_utf8.txt`に、以下の形式で音源と発話テキストを記述してください。
        ```sh
        wavファイル名(拡張子無):発話テキスト　
        ```
    1. 用意した音源が格納されているフォルダを指定して、以下を実行する。
        ```sh
        python3 dataset/preprocess.py --dataset_name homebrew --folder_path ./path/to/wav_folder/ --sampling_rate 44100
        ```
    
## 3. [configs](configs)フォルダ内のjsonを編集
主要なパラメータを説明します。必要であれば編集する。
| 分類  | パラメータ名      | 説明                                                      |
|:-----:|:-----------------:|:---------------------------------------------------------:|
| train | log_interval      | 指定ステップ毎にロスを算出し記録する                      |
| train | eval_interval     | 指定ステップ毎にモデル評価を行う                          |
| train | epochs            | 学習データ全体を学習する回数                              |
| train | batch_size        | 一度のパラメータ更新に使用する学習データ数                |
| train | is_finetune       | ファインチューニングを行うかどうか                        |
| train | finetune_model_dir| ファインチューニング用のcheckpointsが入っているフォルダ   |
| data  | training_files    | 学習用filelistのテキストパス                              |
| data  | validation_files  | 検証用filelistのテキストパス                              |

## 4. 学習
次のコマンドを入力することで、学習を開始する。
> ⚠CUDA Out of Memoryのエラーが出た場合には、config.jsonにてbatch_sizeを小さくする。

-  JSUT Basic5000
    ```sh
    python train.py -c configs/vits2_jsut_nosdp.json -m JSUT_BASIC5000      # no-sdp
    # python train.py -c configs/vits2_jsut_base.json  -m JSUT_BASIC5000    # with sdp:非推奨
    ```

-  ITAコーパス
    ```sh
    python train.py -c configs/vits2_ita_nosdp.json -m ITA_CORPUS   # no-sdp
    # python train.py -c configs/vits2_ita_base.json  -m ITA_CORPUS # with sdp:非推奨
    ```

- 自作データセット
    ```sh
    python train.py -c configs/vits2_homebrew_nosdp.json -m homebrew_dataset    # no-sdp
    # python train.py -c configs/vits2_homebrew_base.json  -m homebrew_dataset  # with sdp:非推奨
    ```

学習経過はターミナルにも表示されるが、tensorboardを用いて確認することで、生成音声の視聴や、スペクトログラム、各ロス遷移を目視で確認することができます。
```sh
tensorboard --logdir logs
```

## 5. 推論
次のコマンドを入力することで、推論を開始する。config.jsonへのパスと、生成器モデルパスを指定する。
```sh
python3 inference.py --config ./path/to/config.json --model_path ./path/to/G_model.pth
```
実行後、テキストを入力することで、音声が生成さされます。音声は自動的に再生され、infer_logsフォルダ（存在しない場合は自動作成）に保存されます。

## 6.ファインチューニング
ファインチューニングを行う場合は、生成器モデルのcheckpointをG_finetune.pth、識別器モデルのcheckpointsをD_finetune.pth、DURモデルのcheckpointをDUR_finetune.pthに名前を変更し、config.jsonで記述しているfinetune_model_dirフォルダ内へと配置する。その後、config.json内のis_finetuneをtrueに変更した後、「4. 学習」のコマンドで学習を開始することで、ファインチューニングを行うことが出来ます。

## 事前学習モデル
- 追加予定

## 参考文献
- https://github.com/p0p4k/vits2_pytorch
- https://github.com/espnet/espnet
