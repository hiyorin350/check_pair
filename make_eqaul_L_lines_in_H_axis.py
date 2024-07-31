import numpy as np
import matplotlib.pyplot as plt
import conversions as con

# 選択肢をユーザーに提示
choice = input("表示したい内容を選んでください:\n1. hsl\n2. mhsl\n選択肢 (1 または 2): ")

# Define L* and H ranges
L_values = np.array([0.78431373,  2.74509804,  5.88235294, 10.98039216, 18.03921569, 27.84313725,
 40.39215686, 56.47058824, 76.07843137])
H_values = np.arange(0, 360, 15)

for S_value in range(0, 101, 10):
    plt.figure(figsize=(10, 6))
    
    for L in L_values:
        Lab_colors = []
        for H in H_values:
            if choice == '2':
                mhsl_image = np.array([[[H, S_value, L]]])
                hsl_image = con.mhsl_to_hsl(mhsl_image)
            elif choice == '1':
                hsl_image = np.array([[[H, S_value, L]]])
            
            rgb_image = con.hsl_to_rgb_non_linear(hsl_image)
            
            Lab_image = con.rgb_to_lab_non_linear(rgb_image)
            L_star = Lab_image[0, 0, 0]
            
            Lab_colors.append((H, L_star))
        
        H_values_plot, L_star_values = zip(*Lab_colors)
        plt.plot(H_values_plot, L_star_values, marker='o', label=f'L={L:.1f}')

    plt.xlabel('H in degrees')
    plt.ylabel('l* in Lab color space')
    plt.title(f'mHSL to Lab Color Space (L*) Relationship\nS = {S_value}')
    plt.legend(title='L in mHSL\n', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'figures/equal_L_lines/evenly_spaced_in_lab_H_axis/mhsl/equal_L_lines_evenly_spaced_ls_in_mhsl_S_{S_value}.png')
    plt.close()