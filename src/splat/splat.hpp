#ifndef SPLAT_HPP
#define SPLAT_HPP

#include <tuple>
#include <vector>

namespace splat {

template <class... Ts>  // Use std::array instead?
using point = std::tuple<Ts...>;

template <class... Ts>
struct weighted_point {
  std::tuple<Ts...> t;
  double w;
};

using point1d = point<double>;
using point2d = point<double, double>;
using point3d = point<double, double, double>;

using wpoint1d = weighted_point<double>;
using wpoint2d = weighted_point<double, double>;
using wpoint3d = weighted_point<double, double, double>;

template <class... Ts>  // Use std::array instead?
using bandwidth = std::tuple<Ts...>;
using bw1d = bandwidth<double>;
using bw2d = bandwidth<double, double>;
using bw3d = bandwidth<double, double, double>;

/* dimension specific paraments */
struct splat_params {
  double min;
  double max;
  double stride;
  size_t n_points;
};

using params1d = std::tuple<splat_params>;  // Use std::array instead?
using params2d = std::tuple<splat_params, splat_params>;
using params3d = std::tuple<splat_params, splat_params, splat_params>;

/* 1D grid splatting with automatically calculated parameters */
std::vector<double> splat_1d(const std::vector<point1d>& points,
                             bw1d bw,
                             size_t nx);

/* 1D weighted grid splatting with automatically calculated parameters */
std::vector<double> wsplat_1d(const std::vector<wpoint1d>& points,
                              bw1d bw,
                              size_t nx);

/* 2D grid splatting with automatically calculated parameters */
std::vector<double> splat_2d(const std::vector<point2d>& points,
                             bw2d bw,
                             size_t nx,
                             size_t ny);

/* 3D grid splatting with automatically calculated parameters */
std::vector<double> splat_3d(const std::vector<point3d>& points,
                             bw3d bw,
                             size_t nx,
                             size_t ny,
                             size_t nz);

/* overload with preset splat parameters */
std::vector<double> splat_3d(const std::vector<point3d>& points,
                             const params3d& params,
                             bw3d bw);

/* 3D weighted grid splatting with automatically calculated parameters */
std::vector<double> wsplat_3d(const std::vector<wpoint3d>& points,
                              bw3d bw,
                              size_t nx,
                              size_t ny,
                              size_t nz);

/* overload with preset splat parameters */
std::vector<double> wsplat_3d(const std::vector<wpoint3d>& points,
                              const params3d& params,
                              bw3d bw);

}  // namespace splat

#endif