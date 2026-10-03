import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter


sphe_path = r"C:\Github\TranslucentSpheroidCompare\TestImages\SpheroidPNG.png"
Bsphe_path = r"C:\Github\TranslucentSpheroidCompare\TestImages\BSpheroidPNG.png"


sphe = cv2.imread(
    sphe_path,
    cv2.IMREAD_UNCHANGED
)

Bsphe = cv2.imread(
    Bsphe_path,
    cv2.IMREAD_UNCHANGED
)


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


def local_contrast(image, mask, window_size=5):

    mean = uniform_filter(
        image,
        size=window_size
    )

    mean_squared = uniform_filter(
        image ** 2,
        size=window_size
    )

    variance = mean_squared - mean ** 2
    variance = np.maximum(variance, 0)

    contrast = np.sqrt(variance)
    contrast[~mask] = 0

    return contrast

normal, normal_mask = prepare_image(sphe)
translucent, translucent_mask = prepare_image(Bsphe)

normal_contrast = local_contrast(
    normal,
    normal_mask,
    window_size=15
)

translucent_contrast = local_contrast(
    translucent,
    translucent_mask,
    window_size=15
)

vmax = max(
    normal_contrast.max(),
    translucent_contrast.max()
)

fig, axes = plt.subplots(
    2,
    2,
    figsize=(10, 10)
)

axes[0, 0].imshow(
    normal,
    cmap="gray"
)

axes[0, 0].set_title(
    "Normal spheroid"
)

axes[0, 0].axis("off")

im1 = axes[0, 1].imshow(
    normal_contrast,
    cmap="jet",
    vmin=0,
    vmax=vmax
)

axes[0, 1].set_title(
    "Normal — local contrast"
)

axes[0, 1].axis("off")

axes[1, 0].imshow(
    translucent,
    cmap="gray"
)

axes[1, 0].set_title(
    "Translucent spheroid"
)

axes[1, 0].axis("off")

im2 = axes[1, 1].imshow(
    translucent_contrast,
    cmap="jet",
    vmin=0,
    vmax=vmax
)

axes[1, 1].set_title(
    "Translucent — local contrast"
)

axes[1, 1].axis("off")

fig.colorbar(
    im2,
    ax=axes,
    shrink=0.8,
    label="Local contrast (SD)"
)


plt.tight_layout()
plt.show()