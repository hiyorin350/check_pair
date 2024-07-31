import cv2
import numpy as np

# 画像のサイズを指定
width, height = 200, 100  # 幅を200ピクセル、高さを100ピクセル
ul_color = (151, 70, 0)  # 左上のRGB値, 赤
ur_color = (44, 107, 44)  # 右上のRGB値、緑
dl_color = (122, 125, 26)  # 左下のRGB値, 黄色
dr_color = (26, 58, 125)  # 右下のRGB値, 赤色
inner_color = (75, 101, 72)  # 内部のRGB値

# 画像を作成し、左右上下で色を分ける
image = np.zeros((height, width, 3), dtype=np.uint8)

# 左上の色を設定
image[:height//2, :width//2] = [ul_color[2], ul_color[1], ul_color[0]]  # OpenCVはBGR順で色を指定

# 右上の色を設定
image[:height//2, width//2:] = [ur_color[2], ur_color[1], ur_color[0]]  # OpenCVはBGR順で色を指定

# 左下の色を設定
image[height//2:, :width//2] = [dl_color[2], dl_color[1], dl_color[0]]  # OpenCVはBGR順で色を指定

# 右下の色を設定
image[height//2:, width//2:] = [dr_color[2], dr_color[1], dr_color[0]]  # OpenCVはBGR順で色を指定

# 内部領域の位置を選択 ('ul' または 'ur')
position = 'ur'  # 'ur' を指定すると右上に内部領域が作成されます

if position == 'ul':
    inner_top_left_x = width // 10  # 左上領域の左上のX座標
    inner_top_left_y = height // 10  # 左上領域の左上のY座標
    inner_bottom_right_x = width // 4  # 左上領域の右下のX座標
    inner_bottom_right_y = height // 4  # 左上領域の右下のY座標
elif position == 'ur':
    inner_top_left_x = width // 2 + width // 10  # 右上領域の左上のX座標
    inner_top_left_y = height // 10  # 右上領域の左上のY座標
    inner_bottom_right_x = width // 2 + width // 4  # 右上領域の右下のX座標
    inner_bottom_right_y = height // 4  # 右上領域の右下のY座標

# 内部領域に色を設定
image[inner_top_left_y:inner_bottom_right_y, inner_top_left_x:inner_bottom_right_x] = [inner_color[2], inner_color[1], inner_color[0]]  # OpenCVはBGR順で色を指定

# 画像をファイルに保存
output_file_path = f'output_image_with_inner_region_{position}.png'
cv2.imwrite(output_file_path, image)

print(f"画像が保存されました: {output_file_path}")
