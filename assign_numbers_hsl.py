import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import conversions

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

# RGBをHSLに変換
hsl_values = np.array([conversions.rgb_to_hsl(np.array([[[row['R'], row['G'], row['B']]]])) for _, row in df.iterrows()])
df[['H', 'S', 'L']] = pd.DataFrame(hsl_values[:, 0, 0, :], index=df.index)

# HSLをHSLtに変換
hsl_t_values = np.array([conversions.hsl_to_mhsl_pixel(row['H'], row['S'], row['L']) for _, row in df.iterrows()])
df[['H', 'S', 'Lt']] = pd.DataFrame(hsl_t_values, index=df.index)

# RGBを16進数のカラーコードに変換する関数
def rgb_to_hex(r, g, b):
    return f'#{int(r):02x}{int(g):02x}{int(b):02x}'

# 各行にカラーコードを追加
df['Color'] = df.apply(lambda row: rgb_to_hex(row['R'], row['G'], row['B']), axis=1)

# 散布図を作成
plt.figure(figsize=(15, 12))
plt.scatter(df['H'], df['Lt'], s=df['Count']/20, c=df['Color'], alpha=0.6)

# ラベルを追加
for i, row in df.iterrows():
    plt.text(row['H'], row['Lt'], f'{int(row["Order"])}', fontsize=9, ha='right')

plt.xlabel('H')
plt.ylabel('Lt')
plt.title('chart26_HSLt')
plt.grid(True)
plt.show()

# 結果をCSVに保存
df.to_csv('/Users/hiyori/check_pair/ordered_hslt_pairs.csv', index=False)
