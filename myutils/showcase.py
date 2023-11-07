import matplotlib.pyplot as plt
import torch
import numpy as np


def showShowcases(showcases, num_rows=1, _multiline=False, _add_label=True):
    # print(len(showcases))
    # if _multiline:
    #     concat_showcases = showcases[0]
    #     for i in range(1, len(showcases)):
    #         concat_showcases = concat_showcases.extend(showcases[i])

    #     showcases = concat_showcases

    num_cols = (len(showcases) + 1) // num_rows

    plt.figure(figsize=(12, 8))

    for i, img_tensor in enumerate(showcases):
        # 添加子图
        plt.subplot(num_rows, num_cols, i + 1)

        # 绘制图像
        plt.imshow(img_tensor, cmap="gray")  # 假设图像是灰度图

        # 添加序号 (a), (b), (c)...
        if _add_label:
            plt.text(
                0.5, -0.12, f"({chr(97 + i)})", fontsize=12, transform=plt.gca().transAxes, horizontalalignment="center"
            )

        # 关闭坐标轴
        plt.axis("off")

    # 调整布局
    plt.tight_layout()

    # 显示图形
    plt.show()
