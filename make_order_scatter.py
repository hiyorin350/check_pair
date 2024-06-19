import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 選択肢をユーザーに提示
choice = input("表示したい内容を選んでください:\n1. Count * Norm\n2. Count * Norm_with_weight\n選択肢 (1 または 2): ")

# CSVファイルの読み込み
csv_file_path = 'csv/converted_rgb_pairs_norm_weight.csv'  # 適切なパスに変更してください
df = pd.read_csv(csv_file_path)

# 同じOrder同士の対を除外
df = df[df['Order1'] != df['Order2']]

# Order1とOrder2が入れ替わっている場合に合算
df['Pair'] = df.apply(lambda row: tuple(sorted((row['Order1'], row['Order2']))), axis=1)

if choice == '1':
    df_grouped = df.groupby('Pair').agg({'Count': 'sum', 'Norm': 'mean'}).reset_index()
    # CountとNormの積を計算して新しい列を追加
    df_grouped['Count_Norm'] = np.floor(df_grouped['Count'] * df_grouped['Norm'])
    value_column = 'Count_Norm'
    title = 'chart26 sigma≒17 kang Count_Norm Scatter Plot'
elif choice == '2':
    df_grouped = df.groupby('Pair').agg({'Count': 'sum', 'Norm_with_weight': 'mean'}).reset_index()
    # CountとNorm_with_weightの積を計算して新しい列を追加
    df_grouped['Count_Norm_with_weight'] = np.floor(df_grouped['Count'] * df_grouped['Norm_with_weight'])
    value_column = 'Count_Norm_with_weight'
    title = 'chart26 sigma≒17 kang Count_Norm_with_weight Scatter Plot'
else:
    print("無効な選択肢です。プログラムを終了します。")
    exit()

# Pair列をOrder1とOrder2に分割
df_grouped[['Order1', 'Order2']] = pd.DataFrame(df_grouped['Pair'].tolist(), index=df_grouped.index)
if choice == '1':
    df_grouped = df_grouped[['Order1', 'Order2', 'Count', 'Norm', value_column]]
elif choice == '2':
    df_grouped = df_grouped[['Order1', 'Order2', 'Count', 'Norm_with_weight', value_column]]

# 散布図を作成
plt.figure(figsize=(15, 12))
plt.scatter(df_grouped['Order1'], df_grouped['Order2'], s=df_grouped[value_column]/20, alpha=0.6)

# 軸のラベルとタイトルを設定
plt.xlabel('Order1')
plt.ylabel('Order2')
plt.title(title)
plt.grid(True)

# 数値を表示（文字を小さく設定）
for i, row in df_grouped.iterrows():
    plt.text(row['Order1'], row['Order2'], f"{int(row[value_column])}", fontsize=7, ha='right')

# グラフを表示
plt.show()

# 結果をCSVに保存
# output_csv_file_path = 'filtered_order_pairs.csv'
# df_grouped.to_csv(output_csv_file_path, index=False)

# データの表示（確認用）
print(df_grouped.head())
