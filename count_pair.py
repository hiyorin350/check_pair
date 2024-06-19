import numpy as np
import cv2
from collections import Counter
import csv
from functions import calculate_color_difference_vectors_with_gaussian_pairing

# 画像の読み込みと色空間の変換
image_bgr = cv2.imread('images/chart26/chart26.ppm')
image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
assert image is not None, "読み込みに失敗しました"

# 画像の高さと幅を取得
height, width, _ = image.shape

Xl, w = calculate_color_difference_vectors_with_gaussian_pairing.calculate_color_difference_vectors_with_gaussian_pairing(image_bgr)

Xl = Xl.reshape((image.shape))

w = w.reshape(height, width)

# シードを設定
seed = 32
np.random.seed(seed)

# sigmaを計算
sigma = (2 / np.pi) * np.sqrt(2 * min(height, width))

# ガウス分布に基づいてランダムなオフセットを生成する関数
def generate_random_offsets(sigma):
    return np.random.normal(0, sigma, 2)

# RGBのペアをカウントするためのCounter
rgb_pair_counter = Counter()

# ノルムを格納する配列を初期化
norm = np.zeros((height, width))
norm_with_weight = np.zeros((height, width))

# 各ペアの情報を保存する辞書
pair_info_dict = {}

# 全画素に対してランダムなオフセットを生成
for y in range(height):
    for x in range(width):
        offset_x, offset_y = generate_random_offsets(sigma=sigma)
        offset_x = int(round(offset_x))
        offset_y = int(round(offset_y))
        
        # 画像範囲内でオフセットを適用
        x2 = min(max(x + offset_x, 0), width - 1)
        y2 = min(max(y + offset_y, 0), height - 1)
        
        rgb1 = tuple(image[y, x])
        rgb2 = tuple(image[y2, x2])
        
        rgb_pair = (rgb1, rgb2)
        
        rgb_pair_counter[rgb_pair] += 1

        # 各ピクセルのノルムを計算して格納
        norm[y, x] = np.linalg.norm([Xl[y, x, 1], Xl[y, x, 2]], ord=2)

        # 重み付きのノルムを計算
        norm_with_weight[y, x] = norm[y, x] * w[y, x]

        # ペア情報を保存
        if rgb_pair not in pair_info_dict:
            pair_info_dict[rgb_pair] = [0, 0, 0]  # [count, norm_sum, norm_with_weight_sum]

        pair_info_dict[rgb_pair][0] += 1
        pair_info_dict[rgb_pair][1] += norm[y, x]
        pair_info_dict[rgb_pair][2] += norm_with_weight[y, x]

# 登場回数が少ない順に並べ替え
sorted_rgb_pairs = sorted(pair_info_dict.items(), key=lambda item: item[1][0])

# CSVファイルに書き込む
output_csv_file_path = 'rgb_pairs_counts_norm_weight.csv'

with open(output_csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["R1", "G1", "B1", "R2", "G2", "B2", "Count", "Norm", "Norm_with_weight"])
    
    for (rgb1, rgb2), (count, norm_sum, norm_with_weight_sum) in sorted_rgb_pairs:
        writer.writerow([rgb1[0], rgb1[1], rgb1[2], rgb2[0], rgb2[1], rgb2[2], count, norm_sum / count, norm_with_weight_sum / count])

print(f"RGB pairs count saved to {output_csv_file_path}")
