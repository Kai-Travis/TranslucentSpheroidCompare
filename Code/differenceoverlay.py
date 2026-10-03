import cv2
import numpy as np
import matplotlib.pyplot as plt

sphe_path = r"C:\Github\TranslucentSpheroidCompare\TestImages\SpheroidPNG.png"
Bsphe_path = r"C:\Github\TranslucentSpheroidCompare\TestImages\BSpheroidPNG.png"

sphe = cv2.imread(sphe_path, cv2.IMREAD_UNCHANGED)
Bsphe = cv2.imread(Bsphe_path, cv2.IMREAD_UNCHANGED)

def prepare_image(img, target_size=250):
    if img.shape[2] == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
        mask = alpha > 0

    else:
        bgr = img
        gray_temp = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        mask = gray_temp > 0

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(
        gray,
        (target_size, target_size),
        interpolation=cv2.INTER_CUBIC
    )

    mask = cv2.resize(
        mask.astype(np.uint8),
        (target_size, target_size),
        interpolation=cv2.INTER_NEAREST
    ).astype(bool)


    return gray.astype(float), mask

normal, normal_mask = prepare_image(sphe)

translucent, translucent_mask = prepare_image(Bsphe)

common_mask = normal_mask & translucent_mask

difference = np.abs(
    translucent - normal
)

difference[~common_mask] = 0

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5)
)

axes[0].imshow(
    normal,
    cmap="gray"
)

axes[0].set_title(
    "Normal spheroid"
)

axes[0].axis("off")

axes[1].imshow(
    translucent,
    cmap="gray"
)

axes[1].set_title(
    "Translucent spheroid"
)

axes[1].axis("off")

im = axes[2].imshow(
    difference,
    cmap="hot",
    vmin=0,
    vmax=255
)

axes[2].set_title(
    "Difference"
)

axes[2].axis("off")

fig.colorbar(
    im,
    ax=axes,
    shrink=0.8,
    label="Absolute intensity difference"
)

plt.tight_layout()
plt.show()