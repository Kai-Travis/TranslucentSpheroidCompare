import cv2
import numpy as np
import matplotlib.pyplot as plt

sphe_path = r"C:\Github\TranslucentSpheroidCompare\TestImages\SpheroidPNG.png"
Bsphe_path = r"C:\Github\TranslucentSpheroidCompare\TestImages\BSpheroidPNG.png"

sphe = cv2.imread(sphe_path, cv2.IMREAD_UNCHANGED)
Bsphe = cv2.imread(Bsphe_path, cv2.IMREAD_UNCHANGED)

def prepare_image(img, target_size=300):

    if img.shape[2] == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
        mask = alpha > 0
    else:
        bgr = img
        gray_temp = cv2.cvtColor( bgr, cv2.COLOR_BGR2GRAY)
        mask = gray_temp > 0

    gray = cv2.cvtColor( bgr, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(gray, (target_size, target_size), interpolation=cv2.INTER_AREA)

    mask = cv2.resize(mask.astype(np.uint8), (target_size, target_size), interpolation=cv2.INTER_NEAREST).astype(bool)

    gray = gray.astype(float)
    gray[~mask] = 0

    return gray


normal = prepare_image(sphe)
translucent = prepare_image(Bsphe)

size = normal.shape[0]

x = np.arange(size)
y = np.arange(size)

X, Y = np.meshgrid(x, y)

z_min = 0
z_max = 255

fig = plt.figure(figsize=(15, 7))

ax1 = fig.add_subplot(
    1,
    2,
    1,
    projection="3d"
)

surface1 = ax1.plot_surface(
    X,
    Y,
    normal,
    cmap="jet",
    vmin = z_min,
    vmax = z_max,
    linewidth=0,
    antialiased=True
)

ax1.set_title(
    "Normal spheroid"
)

ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Intensity")

ax1.set_zlim(z_min, z_max)

ax1.view_init(
    elev=35,
    azim=-60
)

ax2 = fig.add_subplot(
    1,
    2,
    2,
    projection="3d"
)

surface2 = ax2.plot_surface(
    X,
    Y,
    translucent,
    cmap="jet",
    vmin=z_min,
    vmax = z_max,
    linewidth=0,
    antialiased=True
)

ax2.set_title(
    "Translucent spheroid"
)

ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Intensity")

ax2.set_zlim(z_min, z_max)

ax2.view_init(elev=35, azim=-60)

fig.colorbar(
    surface2,
    ax=[ax1, ax2],
    shrink=0.65,
    pad = 0.1,
    label="Grayscale intensity"
)


plt.tight_layout()

plt.show()