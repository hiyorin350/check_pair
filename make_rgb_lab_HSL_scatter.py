import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from conversions import rgb_to_lab, rgb_to_hsl_pixel, hsl_to_mhsl_pixel

# 表示モードの選択をユーザーに提示
display_mode = input("表示したいモードを選んでください:\n1. RGB\n2. Lab\n3. HSL\n4. HSLt\n選択肢 (1, 2, 3 または 4): ")

# CSVファイルを読み込む
csv_file_path = 'csv/rgb_counts.csv'  # 適切なパスに変更してください
df = pd.read_csv(csv_file_path)

# RGBをLabに変換して各行に追加
if display_mode == '2':
    df[['L', 'a', 'b']] = df.apply(lambda row: pd.Series(rgb_to_lab(np.array([row['R'], row['G'], row['B']]))), axis=1)
elif display_mode == '3':
    df[['H', 'S', 'L']] = df.apply(lambda row: pd.Series(rgb_to_hsl_pixel(row['R'], row['G'], row['B'])), axis=1)
elif display_mode == '4':
    df[['H', 'S', 'Lt']] = df.apply(lambda row: pd.Series(hsl_to_mhsl_pixel(*rgb_to_hsl_pixel(row['R'], row['G'], row['B']))), axis=1)

# RGBを16進数のカラーコードに変換する関数
def rgb_to_hex(r, g, b):
    r, g, b = int(r), int(g), int(b)  # 整数に変換
    return f'#{r:02x}{g:02x}{b:02x}'

# 各行にカラーコードを追加
df['Color'] = df.apply(lambda row: rgb_to_hex(row['R'], row['G'], row['B']), axis=1)

# 散布図を作成
plt.figure(figsize=(15, 12))

if display_mode == '1':
    # RGBモードの軸の選択をユーザーに提示
    axis_mode = input("表示したい軸を選んでください:\n1. RG\n2. RB\n選択肢 (1 または 2): ")

    if axis_mode == '1':
        # RG軸の散布図を作成
        plt.scatter(df['R'], df['G'], s=df['Count']/20, c=df['Color'], alpha=0.6)
        plt.xlabel('Red')
        plt.ylabel('Green')
        plt.title('chart26_RGB_RG')
        # ラベルを追加
        for i, row in df.iterrows():
            plt.text(row['R'], row['G'], f'({int(row["R"])},{int(row["G"])},{int(row["B"])}), {row["Count"]}', fontsize=7, ha='right')
    elif axis_mode == '2':
        # RB軸の散布図を作成
        plt.scatter(df['R'], df['B'], s=df['Count']/20, c=df['Color'], alpha=0.6)
        plt.xlabel('Red')
        plt.ylabel('Blue')
        plt.title('chart26_RGB_RB')
        # ラベルを追加
        for i, row in df.iterrows():
            plt.text(row['R'], row['B'], f'({int(row["R"])},{int(row["G"])},{int(row["B"])}), {row["Count"]}', fontsize=7, ha='right')
    else:
        print("無効な選択肢です。プログラムを終了します。")
        exit()
elif display_mode == '2':
    # Labモードの軸の選択をユーザーに提示
    axis_mode = input("表示したい軸を選んでください:\n1. aL\n2. ab\n選択肢 (1 または 2): ")

    if axis_mode == '1':
        # aL軸の散布図を作成
        plt.scatter(df['a'], df['L'], s=df['Count']/20, c=df['Color'], alpha=0.6)
        plt.xlabel('a')
        plt.ylabel('L')
        plt.title('chart26_Lab_aL')
        # ラベルを追加
        for i, row in df.iterrows():
            plt.text(row['a'], row['L'], f'({row["L"]:.1f}, {row["a"]:.1f}, {row["b"]:.1f}), {row["Count"]}', fontsize=7, ha='right')
    elif axis_mode == '2':
        # ab軸の散布図を作成
        plt.scatter(df['a'], df['b'], s=df['Count']/20, c=df['Color'], alpha=0.6)
        plt.xlabel('a')
        plt.ylabel('b')
        plt.title('chart26_Lab_ab')
        # ラベルを追加
        for i, row in df.iterrows():
            plt.text(row['a'], row['b'], f'({row["L"]:.1f}, {row["a"]:.1f}, {row["b"]:.1f}), {row["Count"]}', fontsize=7, ha='right')
    else:
        print("無効な選択肢です。プログラムを終了します。")
        exit()
elif display_mode == '3':
    # HSLモードの軸の選択をユーザーに提示
    axis_mode = input("表示したい軸を選んでください:\n1. HS\n2. HL\n選択肢 (1 または 2): ")

    if axis_mode == '1':
        # HS軸の散布図を作成
        plt.scatter(df['H'], df['S'], s=df['Count']/20, c=df['Color'], alpha=0.6)
        plt.xlabel('Hue')
        plt.ylabel('Saturation')
        plt.title('chart26_HSL_HS')
        # ラベルを追加
        for i, row in df.iterrows():
            plt.text(row['H'], row['S'], f'({row["H"]:.1f}, {row["S"]:.1f}, {row["L"]:.1f}), {row["Count"]}', fontsize=7, ha='right')
    elif axis_mode == '2':
        # HL軸の散布図を作成
        plt.scatter(df['H'], df['L'], s=df['Count']/20, c=df['Color'], alpha=0.6)
        plt.xlabel('Hue')
        plt.ylabel('Lightness')
        plt.title('chart26_HSL_HL')
        # ラベルを追加
        for i, row in df.iterrows():
            plt.text(row['H'], row['L'], f'({row["H"]:.1f}, {row["S"]:.1f}, {row["L"]:.1f}), {row["Count"]}', fontsize=7, ha='right')
    else:
        print("無効な選択肢です。プログラムを終了します。")
        exit()
elif display_mode == '4':
    # HSLtモードの軸の選択をユーザーに提示
    axis_mode = input("表示したい軸を選んでください:\n1. HS\n2. HLt\n選択肢 (1 または 2): ")

    if axis_mode == '1':
        # HS軸の散布図を作成
        plt.scatter(df['H'], df['S'], s=df['Count']/20, c=df['Color'], alpha=0.6)
        plt.xlabel('Hue')
        plt.ylabel('Saturation')
        plt.title('chart26_HSLt_HS')
        # ラベルを追加
        for i, row in df.iterrows():
            plt.text(row['H'], row['S'], f'({row["H"]:.1f}, {row["S"]:.1f}, {row["Lt"]:.1f}), {row["Count"]}', fontsize=7, ha='right')
    elif axis_mode == '2':
        # HLt軸の散布図を作成
        plt.scatter(df['H'], df['Lt'], s=df['Count']/20, c=df['Color'], alpha=0.6)
        plt.xlabel('Hue')
        plt.ylabel('Lt')
        plt.title('chart26_HSLt_HLt')
        # ラベルを追加
        for i, row in df.iterrows():
            plt.text(row['H'], row['Lt'], f'({row["H"]:.1f}, {row["S"]:.1f}, {row["Lt"]:.1f}), {row["Count"]}', fontsize=7, ha='right')
    else:
        print("無効な選択肢です。プログラムを終了します。")
        exit()
else:
    print("無効な選択肢です。プログラムを終了します。")
    exit()

plt.grid(True)
plt.show()
