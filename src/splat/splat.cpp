#include "splat.hpp"

#include <algorithm>
#include <cmath>
#include <tuple>
#include <vector>

namespace splat {

// overloads for the two different point types
template<size_t I, class... Ts>
auto& get_coordinate(const weighted_point<Ts...>& p) {
    return std::get<I>(p.t);
}

template<size_t I, class... Ts>
auto& get_coordinate(const point<Ts...>& p) {
    return std::get<I>(p);
}

template <class... Ts>
const double splat_point(const double x, const point<Ts...>&) {
  return x;
}

template <class... Ts>
const double splat_point(const double x, const weighted_point<Ts...>& p) {
  return x * p.w;
}

template<class... Ts>
auto get_index_sequence(const point<Ts...>&) {
  return std::index_sequence_for<Ts...>{};
}

template <class... Ts>
auto get_index_sequence(const weighted_point<Ts...>&) {
  return std::index_sequence_for<Ts...>{};
}

// grid index for a given value
const size_t get_index(double x, const splat_params& params) {
  if (x < params.min) {
    return 0;
  } else if (x > params.max) {
    return params.n_points - 1;
  } else {
    return static_cast<size_t>((x - params.min) / params.stride);
  }
}

// template functions to generate tuple of index bounds from point and params
struct square_bounds {
  size_t min_i;
  size_t max_i;
};

// grid indices that span the splatting range
square_bounds point_to_bound(double x, const splat_params& params, double bw) {
  double x_min = x - 2 * bw;  // change the magic number
  double x_max = x + 2 * bw;
  return {get_index(x_min, params), get_index(x_max, params)};
}

template <class Point, class Params, class Bandwidth, size_t... Is>
auto make_bound_tuple_impl(const Point& p, const Params& params,
                           const Bandwidth& bw, std::index_sequence<Is...>) {
  return std::make_tuple(point_to_bound(
      get_coordinate<Is>(p), std::get<Is>(params), std::get<Is>(bw))...);
}

template <class Point, class Params, class Bandwidth>
auto make_bound_tuple(const Point& p, const Params& params,
                      const Bandwidth& bw) {
  return make_bound_tuple_impl(p, params, bw, get_index_sequence(p));
}

// templated splatting function
template <size_t... Is, class Tuple>
constexpr size_t prod_index_tuple(const Tuple& t) {
  return (std::get<Is>(t).n_points * ...);
}

template <size_t I, size_t... Is, class Point, class Bounds, class Params,
          class Bandwidth>
void loop_head(std::vector<double>& data, const Point& p, const Bounds& bounds,
               const Params& params, const Bandwidth& bw, size_t offset,
               double parent_dist) {
  const auto& head_bounds = std::get<I>(bounds);
  const auto& head_pos = get_coordinate<I>(p);
  const auto& head_params = std::get<I>(params);
  const auto& head_bandwidth = std::get<I>(bw);

  for (size_t i = head_bounds.min_i; i <= head_bounds.max_i; ++i) {
    double pos = head_params.min + head_params.stride * i;
    double d = (head_pos - pos) / head_bandwidth;
    double cum_dist = parent_dist + d * d;

    if constexpr (sizeof...(Is) > 0) {
      size_t tail_len = prod_index_tuple<Is...>(params);
      loop_head<Is...>(data, p, bounds, params, bw, offset + i * tail_len,
                       cum_dist);
    } else {
      double w = std::exp(-0.5 * cum_dist);
      data[i + offset] += splat_point(w, p);
    }
  }
}

template <class Point, class Bounds, class Params, class Bandwidth,
          size_t... Is>
void loop_impl(std::vector<double>& data, const Point& p, const Bounds& bounds,
               const Params& params, const Bandwidth& bw,
               std::index_sequence<Is...>) {
  loop_head<Is...>(data, p, bounds, params, bw, 0, 0.0);
}

template <class Point, class Params, class Bandwidth>
std::vector<double> splat(const std::vector<Point>& points,
                          const Params& params, const Bandwidth& bw) {
  size_t n_points =
      std::apply([](auto... args) { return (args.n_points * ...); }, params);
  std::vector<double> data(n_points, 0.0);

  // check if points is empty?
  auto is = get_index_sequence(points.front());

  for (const auto& p : points) {
    auto bounds = make_bound_tuple(p, params, bw);
    loop_impl(data, p, bounds, params, bw, is);
  }

  return data;
}

template <size_t I, class Point, class Bandwidth, class Size>
splat_params minmax_tuple(const std::vector<Point>& points, const Bandwidth& bw,
                          const Size& sz) {
  const auto [min_it, max_it] = std::minmax_element(
      points.begin(), points.end(), [](const auto& a, const auto& b) {
        return get_coordinate<I>(a) < get_coordinate<I>(b);
      });
      
  // change the magic number
  double min = get_coordinate<I>(*min_it) - 4 * std::get<I>(bw);
  double max = get_coordinate<I>(*max_it) + 4 * std::get<I>(bw);
  size_t n_points = std::get<I>(sz);
  double stride = (max - min) / n_points;

  return {min, max, stride, n_points};
}

template <class Point, class Bandwidth, class Size, size_t... Is>
auto make_params_impl(const std::vector<Point>& points, const Bandwidth& bw,
                      const Size& sz, std::index_sequence<Is...>) {
  return std::make_tuple(minmax_tuple<Is>(points, bw, sz)...);
}

template <class Point, class Bandwidth, class Size>
auto make_params(const std::vector<Point>& points, const Bandwidth& bw,
                 const Size& sz) {
  auto is = get_index_sequence(points.front());
  return make_params_impl(points, bw, sz, is);
}

/* template specializations for 1-D, 2-D, and 3-D cases in .cpp file to reduce
 * compilation times */
std::vector<double> splat_1d(const std::vector<point1d>& points, bw1d bw,
                             size_t nx) {
  auto params = make_params(points, bw, std::make_tuple(nx));
  return splat(points, params, bw);
}

std::vector<double> wsplat_1d(const std::vector<wpoint1d>& points, bw1d bw,
                              size_t nx) {
  auto params = make_params(points, bw, std::make_tuple(nx));
  return splat(points, params, bw);
}

std::vector<double> splat_2d(const std::vector<point2d>& points, bw2d bw,
                             size_t nx, size_t ny) {
  auto params = make_params(points, bw, std::make_tuple(nx, ny));
  return splat(points, params, bw);
}

std::vector<double> splat_3d(const std::vector<point3d>& points, bw3d bw,
                             size_t nx, size_t ny, size_t nz) {
  auto params = make_params(points, bw, std::make_tuple(nx, ny, nz));
  return splat(points, params, bw);
}

std::vector<double> splat_3d(const std::vector<point3d>& points,
                             const params3d& params,
                             bw3d bw) {
  return splat(points, params, bw);
}

std::vector<double> wsplat_3d(const std::vector<wpoint3d>& points, bw3d bw,
                              size_t nx, size_t ny, size_t nz) {
  auto params = make_params(points, bw, std::make_tuple(nx, ny, nz));
  return splat(points, params, bw);
}

std::vector<double> wsplat_3d(const std::vector<wpoint3d>& points,
                              const params3d& params,
                              bw3d bw) {
  return splat(points, params, bw);
}

}  // namespace splat