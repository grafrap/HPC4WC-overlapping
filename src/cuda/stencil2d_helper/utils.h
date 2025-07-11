#pragma once

#include <ostream>
#include <vector>
#include <cuda_runtime.h>

template <typename T> class Storage3D {
public:
  Storage3D(int x, int y, int z, int nhalo, T value = 0)
      : xsize_(x + 2 * nhalo), ysize_(y + 2 * nhalo), zsize_(z),
        halosize_(nhalo),
        data_((x + 2 * nhalo) * (y + 2 * nhalo) * (z + 2 * nhalo), value),
        d_data_(nullptr) {}

  // Destructor to free device memory
  ~Storage3D() {
    if (d_data_) {
      cudaFree(d_data_);
    }
  }

  T &operator()(int i, int j, int k) {
    return data_[i + j * xsize_ + k * xsize_ * ysize_];
  }

  // GPU memory allocation
  void allocateDevice() {
    size_t total_size = xsize_ * ysize_ * zsize_ * sizeof(T);
    cudaMalloc(&d_data_, total_size);
  }

  // Copy data from host to device
  void copyToDevice() {
    if (d_data_) {
      size_t total_size = xsize_ * ysize_ * zsize_ * sizeof(T);
      cudaMemcpy(d_data_, data_.data(), total_size, cudaMemcpyHostToDevice);
    }
  }

  // Copy data from device to host
  void copyFromDevice() {
    if (d_data_) {
      size_t total_size = xsize_ * ysize_ * zsize_ * sizeof(T);
      cudaMemcpy(data_.data(), d_data_, total_size, cudaMemcpyDeviceToHost);
    }
  }

  // Get device pointer
  T* deviceData() { return d_data_; }
  
  // Get host pointer
  T* data() { return data_.data(); }
  
  // Get total size
  size_t size() const { return xsize_ * ysize_ * zsize_; }

  void writeFile(std::ostream &os) {
    int32_t three = 3;
    int32_t sixtyfour = 64;
    int32_t writehalo = halosize_;
    int32_t writex = xsize_;
    int32_t writey = ysize_;
    int32_t writez = zsize_;

    os.write(reinterpret_cast<const char *>(&three), sizeof(three));
    os.write(reinterpret_cast<const char *>(&sixtyfour), sizeof(sixtyfour));
    os.write(reinterpret_cast<const char *>(&writehalo), sizeof(writehalo));
    os.write(reinterpret_cast<const char *>(&writex), sizeof(writex));
    os.write(reinterpret_cast<const char *>(&writey), sizeof(writey));
    os.write(reinterpret_cast<const char *>(&writez), sizeof(writez));
    for (std::size_t k = 0; k < zsize_; ++k) {
      for (std::size_t j = 0; j < ysize_; ++j) {
        for (std::size_t i = 0; i < xsize_; ++i) {
          os.write(reinterpret_cast<const char *>(&operator()(i, j, k)),
                   sizeof(double));
        }
      }
    }
  }

  void initialize() {
    for (std::size_t k = zsize_ / 4.0; k < 3 * zsize_ / 4.0; ++k) {
      for (std::size_t j = ysize_ / 4.;
           j < 3. / 4. * ysize_; ++j) {
        for (std::size_t i = xsize_ / 4.;
             i < 3. / 4. * xsize_; ++i) {
          operator()(i, j, k) = 1;
        }
      }
    }
  }

  std::size_t xMin() const { return halosize_; }
  std::size_t xMax() const { return xsize_ - halosize_; }
  std::size_t xSize() const { return xsize_; }
  std::size_t yMin() const { return halosize_; }
  std::size_t yMax() const { return ysize_ - halosize_; }
  std::size_t ySize() const { return ysize_; }
  std::size_t zMin() const { return 0; }
  std::size_t zMax() const { return zsize_; }

private:
  int32_t xsize_, ysize_, zsize_, halosize_;
  std::vector<T> data_;
  T* d_data_;  // Device pointer
};

void updateHalo(Storage3D<double> &inField) {
  const int xInterior = inField.xMax() - inField.xMin();
  const int yInterior = inField.yMax() - inField.yMin();

  // bottom edge (without corners)
  for (std::size_t k = 0; k < inField.zMin(); ++k) {
    for (std::size_t j = 0; j < inField.yMin(); ++j) {
      for (std::size_t i = inField.xMin(); i < inField.xMax(); ++i) {
        inField(i, j, k) = inField(i, j + yInterior, k);
      }
    }
  }

  // top edge (without corners)
  for (std::size_t k = 0; k < inField.zMin(); ++k) {
    for (std::size_t j = inField.yMax(); j < inField.ySize(); ++j) {
      for (std::size_t i = inField.xMin(); i < inField.xMax(); ++i) {
        inField(i, j, k) = inField(i, j - yInterior, k);
      }
    }
  }

  // left edge (including corners)
  for (std::size_t k = 0; k < inField.zMin(); ++k) {
    for (std::size_t j = inField.yMin(); j < inField.yMax(); ++j) {
      for (std::size_t i = 0; i < inField.xMin(); ++i) {
        inField(i, j, k) = inField(i + xInterior, j, k);
      }
    }
  }

  // right edge (including corners)
  for (std::size_t k = 0; k < inField.zMin(); ++k) {
    for (std::size_t j = inField.yMin(); j < inField.yMax(); ++j) {
      for (std::size_t i = inField.xMax(); i < inField.xSize(); ++i) {
        inField(i, j, k) = inField(i - xInterior, j, k);
      }
    }
  }
}