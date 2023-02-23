# Foreground Reconstruction

- Build and install Image-Adaptive-3DLUT
  - For CUDA 11.x, run `sh setup.sh` at `LUT/trilinear_cpp` and ignore the failing message

- For demo, see jupyter notebooks at root
  - Follow `polynomial_color_transfer.ipynb` to turn the polynomial color transfer coefficient `demo/poly.npy` into 3D-LUT and apply the polynomial color transfer to `demo/img.jpg`
  - Follow `lut_color_transfer.ipynb` to apply the 3D-LUT color transfer to the image