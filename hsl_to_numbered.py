import pandas as pd
import numpy as np

def rgb_to_hsl(r, g, b):
    """
    RGBをHSLに変換
    """
    R = r / 255.0
    G = g / 255.0
    B = b / 255.0

    max_color = np.maximum(np.maximum(R, G), B)
    min_color = np.minimum(np.minimum(R, G), B)

    L = (max_color + min_color) / 2.0

    delta = max_color - min_color
    S = np.zeros_like(L)
    
    # 彩度の計算
    S[delta != 0] = delta[delta != 0] / (1 - np.abs(2 * L[delta != 0] - 1))

    H = np.zeros_like(L)
    # 色相の計算
    # Rが最大値
    idx = (max_color == R) & (delta != 0)
    H[idx] = 60 * (((G[idx] - B[idx]) / delta[idx]) % 6)

    # Gが最大値
    idx = (max_color == G) & (delta != 0)
    H[idx] = 60 * (((B[idx]) - (R[idx])) / delta[idx] + 2)

    # Bが最大値
    idx = (max_color == B) & (delta != 0)
    H[idx] = 60 * (((R[idx]) - (G[idx])) / delta[idx] + 4)

    # 彩度と輝度をパーセンテージに変換
    S = S * 100
    L = L * 100

    return np.stack([H, S, L], axis=-1)

def hsl_to_cartesian(hsl):
    """
    HSL色空間で表された画像を直交座標系に変換する。
    param
    hsl:h(degree), s[0,100], l[0,100]
    return
    xyl:x[0,100], y[0,100], l[0,100]
    """
    h_rad = np.deg2rad(hsl[0])  # 色相をラジアンに変換
    s = hsl[1]  # 彩度
    l = hsl[2]  # 輝度
    
    x = s * np.cos(h_rad)  # x
    y = s * np.sin(h_rad)  # y
    
    return np.array([x, y, l])

# CSVファイルを読み込む
input_csv_file_path = 'csv/rgb_pairs_counts_norm_weight.csv'  # 適切なパスに変更してください
output_csv_file_path = 'csv/hsl_pairs_counts_norm_weight.csv'

sigma_weight = 5  # 明度差重みのパラメータ

df = pd.read_csv(input_csv_file_path)

# 各行のRGB値をHSL値に変換
df[['H1', 'S1', 'L1']] = df.apply(lambda row: pd.Series(rgb_to_hsl(row['R1'], row['G1'], row['B1'])), axis=1)
df[['H2', 'S2', 'L2']] = df.apply(lambda row: pd.Series(rgb_to_hsl(row['R2'], row['G2'], row['B2'])), axis=1)

# HSL値を小数点以下3桁までにフォーマット
df['H1'] = df['H1'].round(3)
df['S1'] = df['S1'].round(3)
df['L1'] = df['L1'].round(3)
df['H2'] = df['H2'].round(3)
df['S2'] = df['S2'].round(3)
df['L2'] = df['L2'].round(3)

# HSLを直交座標系に変換
xyl_1 = np.array(df.apply(lambda row: hsl_to_cartesian(np.array([row['H1'], row['S1'], row['L1']])), axis=1).tolist())
xyl_2 = np.array(df.apply(lambda row: hsl_to_cartesian(np.array([row['H2'], row['S2'], row['L2']])), axis=1).tolist())

# ノルムを算出 (xとyのみ使用)
norm = np.linalg.norm(xyl_1[:, :2] - xyl_2[:, :2], axis=1)

# ノルムを新しい列としてDataFrameに追加
df['Norm'] = norm

# 重みwを計算
w = np.exp(-np.square(xyl_1[:, 2] - xyl_2[:, 2]) / (2 * np.square(sigma_weight)))

norm_with_weight = norm * w

df['Norm_with_weight'] = norm_with_weight

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


# 順番を取得する関数
def get_order(rgb_tuple):
    return order_mapping.get(rgb_tuple, None)

# RGB値を順番に変換
df['Order1'] = df.apply(lambda row: get_order((row['R1'], row['G1'], row['B1'])), axis=1)
df['Order2'] = df.apply(lambda row: get_order((row['R2'], row['G2'], row['B2'])), axis=1)

# 必要な列だけを保持して保存
df_result = df[['Order1', 'Order2', 'Count', 'Norm', 'Norm_with_weight', 'H1', 'S1', 'L1', 'H2', 'S2', 'L2']]
df_result.to_csv(output_csv_file_path, index=False)

print(f"HSL pairs count saved to {output_csv_file_path}")
