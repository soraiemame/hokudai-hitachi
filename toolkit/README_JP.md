# ツールキット
- テストケースジェネレータ
- ジャッジ
- サンプルコード
- ビジュアライザ
が付属しています。

# 実行環境
以下が利用できる環境が必要:
- C++17対応コンパイラ
- Python(Python 3.8以降)
  - `pip`で`requirements.txt`に記載されたパッケージがインストールできる必要あり(**準備**の項を参照)
- bash(5.0.17(1)で動作確認済み)

## 環境例

### Cygwin(3.4.2-1, setup ver.:2.923)
以下のパッケージを追加する:
- gcc-g++ (10.2.0-1で動作確認済み)
- python38-devel(3.8.9-2で動作確認済み)
- python38-numpy(1.21.4-1で動作確認済み)

### Ubuntu 20.04(Docker上のubuntu:20.04)
以下のパッケージを追加する:
- `g++` (4:9.3.0-1ubuntu2 amd64で動作確認済み)
- `python3-pip` (20.0.2-5ubuntu1.6 allで動作確認済み)
- `python3-numpy` (1:1.17.4-5ubuntu3.1 amd64で動作確認済み)
```bash
sudo apt install g++ python3-pip python3-numpy
```

### Ubuntu 22.04(Docker上のubuntu:22.04)
以下のパッケージを追加する:
- `g++` (4:11.2.0-1ubuntu1 amd64で動作確認済み)
- `python3-pip` (22.0.2+dfsg-1 allで動作確認済み)
- `python3-numpy` (1:1.21.5-1ubuntu22.04.1 amd64で動作確認済み)
```bash
sudo apt install g++ python3-pip python3-numpy
```


# 準備
1. `./build.sh`
2. `python3 -m pip install -r requirements.txt`

# 実行

## テストケース生成
```bash
cd generator

# パラメータファイル（初期設定）生成
./random_world.py -g config.toml 

# 必要ならば config.toml を書き換える。各設定項目に説明が付いている。
# A問題用のテストケースを生成するなら、type = "A", Bならば type = "B" と書き換える。

# 設定ファイル(config.toml)を読み込みテストケース(testcase.txt)生成
./random_world.py -c config.toml > testcase.txt 
```

## ジャッジの実行
`judge.sh`に作成した解答プログラムを実行するコマンドを与えることで手元でジャッジを実行できる。使い方は下記の通り:
```bash
./judge.sh テストケースファイル ログ出力先 解答プログラムを実行するコマンド...
```

例1(実行可能ファイル(例えば`a.out`とする)を作成し呼び出す場合):
```bash
./judge.sh generator/testcase.txt visualizer/default.json ./a.out
```

例2(Pythonのコード(例えば`answer.py`とする)を呼び出す場合):
```bash
./judge.sh generator/testcase.txt visualizer/default.json python3 answer.py
```

例3(入力を無視してテキストファイル(例えば`answer.txt`とする)を出力する場合):
```bash
./judge.sh generator/testcase.txt visualizer/default.json sh -c "cat > /dev/null|cat answer.txt"
```

## サンプルコードの実行

A問題
```bash
./judge.sh generator/testcase.txt visualizer/default.json sample/sample_A
```
(テストケース`generator/testcase.txt`は問題A用に生成する)

B問題
```bash
./judge.sh generator/testcase.txt visualizer/default.json sample/sample_B
```
(テストケース`generator/testcase.txt`は問題B用に生成する)

## ビジュアライザ (`visualizer/index.html`)

`visualizer/index.html`を参照。出力されたログファイル(`judge.sh`の第二引数)を利用して可視化する。末尾にクエリ`?en`を与えると英語版が表示される。(`visualizer/index.html?en`)
