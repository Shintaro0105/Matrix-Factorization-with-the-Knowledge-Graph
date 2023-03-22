# Matrix-Factorization-with-the-Knowledge-Graph (卒業研究)
## はじめに

DemoではDouban Movieデータセットにおけるユーザをランダムに30％の人数抽出した縮小データセットにて実行している．

データセットの取得リンク：https://github.com/librahu/HIN-Datasets-for-Recommendation-and-Network-Embedding.git

## 必要環境

必要ライブラリは以下．

- Python
- scipy
- numpy
- pandas
- scikit-learn

また，ItemKNN，MostPop，Randomにおいて，以下のライブラリを用いた．

- [Case Recommender](https://github.com/caserec/CaseRecommender.git)

評価関数とその他の比較手法のライブラリは以下．

- [Recommenders](https://github.com/microsoft/recommenders.git)

グラフ埋め込みライブラリは以下．

- [DeepWalk](https://github.com/phanein/deepwalk.git)