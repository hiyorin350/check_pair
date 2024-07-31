import pandas as pd
import numpy as np

# hsl_to_mhsl関数の定義
def hsl_to_mhsl_pixel(H, S, L):
    R, E, G, C, B, M = 0.30, 0.66, 0.59, 0.64, 0.12, 0.26

    # Hの値に基づいてqとtを計算
    q = (H / 60).astype(int)
    t = H % 60

    a = [R, E, G, C, B, M, R]

    # alpha, l_fun_smax, l_funの計算
    alpha = np.take(a, q + 1) * (t / 60.0) + np.take(a, q) * (60.0 - t) / 60.0
    l_fun_smax = -np.log2(alpha)
    l_fun = l_fun_smax * (S / 100.0) + (1.0 - (S / 100.0))

    # l_tildaの計算とh_tilda, s_tildaの設定
    l_tilda = 100 * (L / 100.0) ** l_fun
    h_tilda = H
    s_tilda = S

    # 修正されたHSL値を返す
    return h_tilda, s_tilda, l_tilda

# CSVファイルを読み込む
csv_file_path = 'csv/hsl_pairs_counts_norm_weight.csv'  # 適切なパスに変更してください
df = pd.read_csv(csv_file_path)

# HSL値を修正HSL値に変換
df[['H1', 'S1', 'Lt1']] = df.apply(lambda row: pd.Series(hsl_to_mhsl_pixel(row['H1'], row['S1'], row['L1'])), axis=1)
df[['H2', 'S2', 'Lt2']] = df.apply(lambda row: pd.Series(hsl_to_mhsl_pixel(row['H2'], row['S2'], row['L2'])), axis=1)

# Ltを小数点以下第3位までに丸める
df['Lt1'] = df['Lt1'].round(3)
df['Lt2'] = df['Lt2'].round(3)

# 必要な列だけを保持して保存
output_df = df[['Order1', 'Order2', 'Count', 'Norm', 'Norm_with_weight', 'H1', 'S1', 'Lt1', 'H2', 'S2', 'Lt2']]
output_csv_file_path = 'csv/mhsl_pairs_counts_norm_weight_test.csv'  # 適切なパスに変更してください
output_df.to_csv(output_csv_file_path, index=False)

print(f"Transformed HSL values saved to {output_csv_file_path}")
