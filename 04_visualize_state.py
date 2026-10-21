import os
import numpy as np
import matplotlib.pyplot as plt


# 저장된 Visual Sculpting state 불러오기
data = np.load("outputs/state/state.npz")

depth = data["depth"]
grad_x = data["grad_x"]
grad_y = data["grad_y"]

grad_mag = np.sqrt(
    grad_x ** 2 + grad_y ** 2
)

os.makedirs("outputs/state/images", exist_ok=True)


# ----------------------------------------
# Depth
# ----------------------------------------
plt.figure(figsize=(6, 6))
plt.imshow(depth)
plt.colorbar(label="Depth (m)")
plt.title("Dense Depth Map")
plt.tight_layout()
plt.savefig(
    "outputs/state/images/depth.png",
    dpi=200,
)
plt.close()


# ----------------------------------------
# Gradient X
# ----------------------------------------
plt.figure(figsize=(6, 6))
plt.imshow(grad_x)
plt.colorbar(label="dD/dx")
plt.title("Depth Gradient X")
plt.tight_layout()
plt.savefig(
    "outputs/state/images/gradient_x.png",
    dpi=200,
)
plt.close()


# ----------------------------------------
# Gradient Y
# ----------------------------------------
plt.figure(figsize=(6, 6))
plt.imshow(grad_y)
plt.colorbar(label="dD/dy")
plt.title("Depth Gradient Y")
plt.tight_layout()
plt.savefig(
    "outputs/state/images/gradient_y.png",
    dpi=200,
)
plt.close()


# ----------------------------------------
# Gradient magnitude
# ----------------------------------------
plt.figure(figsize=(6, 6))
plt.imshow(grad_mag)
plt.colorbar(label="Gradient magnitude")
plt.title("Spatial Gradient Magnitude")
plt.tight_layout()
plt.savefig(
    "outputs/state/images/gradient_magnitude.png",
    dpi=200,
)
plt.close()


print("시각화 완료")
print("outputs/state/images/")