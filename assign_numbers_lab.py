import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2

def rgb_to_lab(image):
    """
    RGBからL*a*b*色空間への変換を行う。
    imageは[0, 255]の範囲の値を持つ3次元のNumPy配列（画像）。
    """
    # RGBを[0, 1]の範囲に正規化
    rgb = image / 255.0
    
    # sRGBからリニアRGBへの変換
    def gamma_correction(channel):
        return np.where(channel > 0.04045, ((channel + 0.055) / 1.055) ** 2.4, channel / 12.92)
    
    rgb_linear = gamma_correction(rgb)
    
    # リニアRGBからXYZへの変換
    mat_rgb_to_xyz = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    xyz = np.dot(rgb_linear, mat_rgb_to_xyz.T)
    
    # XYZからL*a*b*への変換
    def xyz_to_lab(t):
        delta = 6/29
        return np.where(t > delta ** 3, t ** (1/3), t / (3 * delta ** 2) + 4/29)
    
    xyz_ref_white = np.array([0.95047, 1.00000, 1.08883])
    xyz_normalized = xyz / xyz_ref_white
    
    L = 116 * xyz_to_lab(xyz_normalized[..., 1]) - 16
    a = 500 * (xyz_to_lab(xyz_normalized[..., 0]) - xyz_to_lab(xyz_normalized[..., 1]))
    b = 200 * (xyz_to_lab(xyz_normalized[..., 1]) - xyz_to_lab(xyz_normalized[..., 2]))
    
    return np.stack([L, a, b], axis=-1)

# CSVファイルを読み込む
csv_file_path = 'csv/rgb_counts.csv'  # 適切なパスに変更してください
df = pd.read_csv(csv_file_path)

# RGB値を整数に変換
df['R'] = df['R'].astype(int)
df['G'] = df['G'].astype(int)
df['B'] = df['B'].astype(int)

# 順番を決めるマッピング
order_mapping = {
    (173, 82, 119): 1,
    (194, 106, 132): 2,
    (211, 124, 143): 3,
    (233, 107, 108): 4,
    (245, 125, 127): 5,
    (247, 152, 156): 6,
    (239, 177, 189): 7,
    (82, 79, 63): 8,
    (87, 82, 84): 9,
    (99, 90, 74): 10,
    (104, 92, 102): 11,
    (112, 100, 87): 12,
    (120, 109, 104): 13,
    (129, 120, 101): 14,
    (131, 115, 94): 15,
    (138, 123, 133): 16,
    (147, 136, 110): 17,
    (156, 140, 123): 18,
    (156, 156, 132): 19,
    (163, 152, 177): 20,
    (165, 140, 132): 21,
    (165, 148, 123): 22,
    (173, 160, 127): 23,
    (182, 171, 176): 24,
    (187, 179, 149): 25,
    (203, 189, 203): 26,
    (206, 199, 177): 27,
    (222, 222, 206): 28,
    (239, 222, 222): 29,
    (239, 225, 192): 30,
    (239, 239, 222): 31,
    (255, 255, 255): 32
}

order_mapping_ye_bl_3 = {
    (173, 82, 119): 1,
    (194, 106, 132): 2,
    (211, 124, 143): 3,
    (233, 107, 108): 4,
    (245, 125, 127): 5,
    (247, 152, 156): 6,
    (239, 177, 189): 7,
    (82, 79, 63): 8,
    (87, 82, 84): 9,
    (99, 90, 74): 10,
    (104, 92, 102): 11,
    (112, 100, 87): 12,
    (120, 109, 104): 13,
    (129, 120, 101): 14,
    (131, 115, 94): 15,
    (138, 123, 133): 16,
    (147, 136, 110): 17,
    (156, 140, 123): 18,
    (156, 156, 132): 19,
    (163, 152, 177): 20,
    (165, 140, 132): 21,
    (165, 148, 123): 22,
    (173, 160, 127): 23,
    (182, 171, 176): 24,
    (187, 179, 149): 25,
    (203, 189, 203): 26,
    (206, 199, 177): 27,
    (222, 222, 206): 28,
    (239, 222, 222): 29,
    (239, 225, 192): 30,
    (239, 239, 222): 31,
    (255, 255, 255): 32,
    (122, 125, 26): 33,
    (26, 58, 125): 34
}

# 指定された順番の色に番号を振る
df['Order'] = df.apply(lambda row: order_mapping.get((row['R'], row['G'], row['B']), None), axis=1)

# 指定されていない色に対して番号を振る
remaining_df = df[df['Order'].isna()]
remaining_df = remaining_df.assign(RGB_Sum=remaining_df['R'] + remaining_df['G'] + remaining_df['B'])
remaining_df = remaining_df.sort_values(by='RGB_Sum')
remaining_df['Order'] = range(len(order_mapping) + 1, len(df) + 1)

# 結果を統合
df.update(remaining_df)
df = df.sort_values(by='Order')

# RGBをLabに変換
lab_values = np.array([rgb_to_lab(np.array([[[row['R'], row['G'], row['B']]]])) for _, row in df.iterrows()])
df[['L', 'a', 'b']] = pd.DataFrame(lab_values[:, 0, 0, :], index=df.index)

# RGBを16進数のカラーコードに変換する関数
def rgb_to_hex(r, g, b):
    return f'#{int(r):02x}{int(g):02x}{int(b):02x}'

# 各行にカラーコードを追加
df['Color'] = df.apply(lambda row: rgb_to_hex(row['R'], row['G'], row['B']), axis=1)

# 散布図を作成
plt.figure(figsize=(15, 12))
plt.scatter(df['a'], df['L'], s=df['Count']/20, c=df['Color'], alpha=0.6)

# ラベルを追加
for i, row in df.iterrows():
    plt.text(row['a'], row['L'], f'{int(row["Order"])}', fontsize=9, ha='right')

plt.xlabel('a')
plt.ylabel('L')
plt.title('chart26_Lab')
plt.grid(True)
plt.show()

# 結果をCSVに保存
# df.to_csv('/Users/hiyori/check_pair/ordered_lab_pairs.csv', index=False)
