import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルを読み込む
csv_file_path = 'rgb_counts.csv'  # 適切なパスに変更してください
df = pd.read_csv(csv_file_path)

# RGBを16進数のカラーコードに変換する関数
def rgb_to_hex(r, g, b):
    return f'#{r:02x}{g:02x}{b:02x}'

# 各行にカラーコードを追加
df['Color'] = df.apply(lambda row: rgb_to_hex(row['R'], row['G'], row['B']), axis=1)

# 散布図を作成
plt.figure(figsize=(15, 12))
plt.scatter(df['R'], df['G'], s=df['Count']/20, c=df['Color'], alpha=0.6)

# ラベルを追加
for i, row in df.iterrows():
    plt.text(row['R'], row['G'], f'({row["R"]},{row["G"]},{row["B"]}), {row["Count"]}', fontsize=9, ha='right')

plt.xlabel('Red')
plt.ylabel('Green')
plt.title('chart26_RGB')
plt.grid(True)
plt.show()
