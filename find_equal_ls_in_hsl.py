import numpy as np
import conversions as con

# L*の値を0.1から0.9の範囲で生成
ls_values = np.arange(0.1, 1.0, 0.1) * 100
a = np.zeros_like(ls_values)
b = np.zeros_like(ls_values)

# Lab配列を作成
lab = np.stack((ls_values, a, b), axis=1)
lab = lab.reshape(1,9,3)
# print(lab.shape)

# LabからRGB、そしてHSLへ変換
rgb_values = con.lab_to_rgb_non_linear(lab)
hsl_values = con.rgb_to_hsl_non_linear(rgb_values)
mhsl_values = con.hsl_to_mhsl(hsl_values)

print("Lab values:")
print(lab)
print("hsl values:")
print(hsl_values)
print("mhsl values:")
print(mhsl_values)
print("rgb values(Y):")
print(rgb_values[0,:,0] * (100/255))
