import pandas as pd

# CSVファイルの読み込み
csv_file_path = '/Users/hiyori/check_pair/rgb_pairs_counts_norm_weight.csv'  # 適切なパスに変更してください
df = pd.read_csv(csv_file_path)

# RGB値とそれに対応する順番のマッピングを作成
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

# 順番を取得する関数
def get_order(rgb_tuple):
    return order_mapping.get(rgb_tuple, None)

# RGB値を順番に変換
df['Order1'] = df.apply(lambda row: get_order((row['R1'], row['G1'], row['B1'])), axis=1)
df['Order2'] = df.apply(lambda row: get_order((row['R2'], row['G2'], row['B2'])), axis=1)

# 必要な列のみを選択して保存
df_result = df[['Order1', 'Order2', 'Count', 'Norm', 'Norm_with_weight']]
df_result.to_csv('/Users/hiyori/check_pair/converted_rgb_pairs_norm_weight.csv', index=False)
