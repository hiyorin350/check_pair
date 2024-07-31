import numpy as np
import cv2
from collections import Counter

# 画像の読み込みと色空間の変換
image = cv2.cvtColor(cv2.imread('images/chart26_yellow_blue_3.ppm'), cv2.COLOR_BGR2RGB)
assert image is not None, "読み込みに失敗しました"

# 画像の高さと幅を取得
height, width, _ = image.shape

# 各ピクセルの色情報を整数としてエンコード
s = 256 * 256 * image[:,:,0] + 256 * image[:,:,1] + image[:,:,2]

# ユニークな値を取得し、その出現回数をカウント
unique_values, counts = np.unique(s, return_counts=True)

# ユニークな値からRGB成分を抽出
r = unique_values // 65536
g = (unique_values % 65536) // 256
b = unique_values % 256

# RGB成分を行列にまとめる
rgb_matrix = np.vstack((r, g, b, counts)).T

# 行列として出力
print("RGB Matrix with Counts:")
print("R, G, B, Count")
print(rgb_matrix)
