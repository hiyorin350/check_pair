import pandas as pd

# CSVファイルのパスを指定
csv_file_path = '/Users/hiyori/check_pair/rgb_pairs_counts.csv'

# CSVファイルを読み込む
df = pd.read_csv(csv_file_path)

# データフレームの確認
print(df)

# 表として表示するための関数
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# RGB値をカラーコードに変換する関数
def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])

# グラフの描画
fig, ax = plt.subplots(figsize=(12, 8))

# 各行に対してRGBの色を表示
for index, row in df.iterrows():
    rgb1_hex = rgb_to_hex((row['R1'], row['G1'], row['B1']))
    rgb2_hex = rgb_to_hex((row['R2'], row['G2'], row['B2']))
    ax.add_patch(mpatches.Rectangle((0, index), 1, 0.5, color=rgb1_hex))
    ax.add_patch(mpatches.Rectangle((1, index), 1, 0.5, color=rgb2_hex))
    ax.text(2.5, index + 0.25, row['Count'], va='center', ha='left')

# 軸の非表示
ax.axis('off')

# 凡例の作成
unique_rgbs = df[['R1', 'G1', 'B1']].drop_duplicates().apply(lambda row: (row['R1'], row['G1'], row['B1']), axis=1)
legend_patches = [mpatches.Patch(color=rgb_to_hex(rgb), label=f"RGB: {rgb}") for rgb in unique_rgbs]
plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')

plt.show()
