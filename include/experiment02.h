#define PROGRAM_NAME "experiment02"

#define FLAGS_CASES                                                            \
  FLAG_CASE(string, logs_dir, "./logs/", "Logs save directory")                \
  FLAG_CASE(uint64, nframes, 10,                                               \
            "Number of frames considered for initialization")
#define ARGS_CASES ARG_CASE(dataset_dir)

// STL
#include <algorithm>
#include <cmath>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

// Boost
#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>

// Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

// Glog
#include <glog/logging.h>

#include "imu_preintegration.h"
#include "io.h"
#include "methods.h"

#include "util/args.h"
#include "util/csv.h"
#include "util/timer.h"

using namespace std;
namespace fs = boost::filesystem;

using Trajectory = std::vector<io::trajectory_t<double>>;
using Groundtruth = std::vector<io::state_t>;
using ImuData = io::ImuData;

// IMU parameters
// EuRoC
const double rate = 200.;
const double dt = 1. / rate;
const double ng = 1.7e-4;
const double na = 2.e-3;

Eigen::Isometry3d Tcb;

struct evaluation_t {
  evaluation_t(const std::uint64_t execution_time, const double scale_error,
               const double gyro_bias_error, const double gyro_bias_error2,
               const double acc_bias_error, double acc_bias_error2,
               const double gravity_error)
      : execution_time(execution_time), scale_error(scale_error),
        gyro_bias_error(gyro_bias_error), gyro_bias_error2(gyro_bias_error2),
        acc_bias_error(acc_bias_error), acc_bias_error2(acc_bias_error2),
        gravity_error(gravity_error) {}

  std::uint64_t execution_time; // nanoseconds
  double scale_error;           // percent
  double gyro_bias_error;       // percent
  double gyro_bias_error2;      // degress
  double acc_bias_error;        // percent
  double acc_bias_error2;       // degrees
  double gravity_error;         // degrees
};

struct method_result {
  string method_name;
  vector<ResultType> results;
};

void ValidateArgs() { CHECK(fs::is_directory(ARGS_dataset_dir)); }

void ValidateFlags() {
  fs::create_directories(FLAGS_logs_dir);
  CHECK_GT(FLAGS_nframes, 4);
}

Trajectory read_file_TUM(const std::string &path) {
  std::ifstream input(path);

  Trajectory trajectory;
  for (std::string line; std::getline(input, line);) {
    if (line.empty() || line.front() == '#')
      continue;

    std::istringstream iss(line);
    double timestamp, tx, ty, tz, qw, qx, qy, qz;
    if (iss >> timestamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw) {
      io::pose_t pose(tx, ty, tz, qw, qx, qy, qz);
      trajectory.emplace_back(timestamp, pose);
    }
  }
  return trajectory;
}

Trajectory::const_iterator next(Trajectory::const_iterator i,
                                Trajectory::const_iterator j, const double dt) {
  if (i == j) {
    LOG(WARNING) << "Already at the end...";
    return i;
  }

  const double t = i->timestamp + dt;
  Trajectory::const_iterator it = std::upper_bound(
      i, j, t, [](const double lhs, const io::trajectory_t<double> &rhs) {
        return lhs < rhs.timestamp;
      });
  if (it == i)
    return i;
  if (it == j)
    return j;
  Trajectory::const_iterator it_ = std::next(it, -1);
  if ((it->timestamp - t) > (t - it_->timestamp))
    return it_;
  else
    return it;
}

ImuData::const_iterator start_imu(ImuData::const_iterator i,
                                  ImuData::const_iterator j,
                                  io::timestamp_t t) {
  ImuData::const_iterator it = std::upper_bound(
      i, j, t, [](const io::timestamp_t lhs, const io::imu_data_t &rhs) {
        return lhs < rhs.timestamp;
      });
  if (it == i)
    return i;
  if (it == j)
    return j;
  ImuData::const_iterator it_ = std::next(it, -1);
  if ((it->timestamp - t) > (t - it_->timestamp))
    return it_;
  else
    return it;
}

Trajectory::const_iterator start(const Trajectory &trajectory,
                                 const io::ImuData &imu_data) {
  Trajectory::const_iterator i = trajectory.cbegin();
  Trajectory::const_iterator i_ = i;
  while (i != trajectory.cend()) {
    Eigen::Vector3d avgA;
    avgA.setZero();

    io::ImuData::const_iterator it = imu_data.cbegin();
    for (unsigned n = 0; n < FLAGS_nframes; ++n) {
      it = start_imu(it, imu_data.cend(),
                     static_cast<io::timestamp_t>(i->timestamp * 1e9));
      CHECK(it != imu_data.cend());

      // Trajectory::const_iterator j = next(i, trajectory.cend(), 0.25); // 4
      // Hz
      Trajectory::const_iterator j = std::next(i, 1);
      CHECK(j != trajectory.cend());

      std::shared_ptr<IMU::Preintegrated> pInt =
          std::make_shared<IMU::Preintegrated>(Eigen::Vector3d::Zero(),
                                               Eigen::Vector3d::Zero());
      while (it != imu_data.cend() &&
             std::abs(it->timestamp * 1e-9 - j->timestamp) > 0.0025) {
        const Eigen::Vector3d w(it->w_x, it->w_y, it->w_z);
        const Eigen::Vector3d a(it->a_x, it->a_y, it->a_z);
        pInt->IntegrateNewMeasurement(w, a, dt);
        std::advance(it, 1);
      }
      CHECK(it != imu_data.cend());

      avgA += pInt->dV / pInt->dT;
      i = j;
    }

    avgA /= static_cast<double>(FLAGS_nframes);
    const double avgA_error =
        std::abs(avgA.norm() - IMU::GRAVITY_MAGNITUDE) / IMU::GRAVITY_MAGNITUDE;
    LOG(INFO) << "Average acceleration: " << 100. * avgA_error;
    if (avgA_error > 5e-3)
      break;

    i = next(i_, trajectory.cend(), 0.5);
    i_ = i;
  }
  CHECK(i != trajectory.cend());

  return i_;
}

Groundtruth::const_iterator find_closest(Groundtruth::const_iterator i,
                                         Groundtruth::const_iterator j,
                                         io::timestamp_t t) {
  Groundtruth::const_iterator it = std::upper_bound(
      i, j, t, [](const io::timestamp_t lhs, const io::state_t &rhs) {
        return lhs < rhs.timestamp;
      });
  if (it == i)
    return i;
  if (it == j)
    return j;
  Groundtruth::const_iterator it_ = std::next(it, -1);
  if ((it->timestamp - t) > (t - it_->timestamp))
    return it_;
  else
    return it;
}

Eigen::Isometry3d compute_scale(const InputType &input,
                                const Groundtruth &groundtruth,
                                double &scale_factor) {

  io::Trajectory trajectory;
  trajectory.emplace_back(input.front().t1, input.front().T1);
  for (const input_t &entry : input)
    trajectory.emplace_back(entry.t2, entry.T2);

  std::vector<std::pair<unsigned, unsigned>> pairs;
  for (io::Trajectory::const_iterator it = trajectory.cbegin();
       it != trajectory.cend(); ++it) {
    Groundtruth::const_iterator jt = std::upper_bound(
        groundtruth.cbegin(), groundtruth.cend(), it->timestamp,
        [](const io::timestamp_t lhs, const io::state_t &rhs) {
          return lhs < rhs.timestamp;
        });
    if (jt->timestamp - it->timestamp > 2500000) {
      if (jt == groundtruth.cbegin())
        continue;
      std::advance(jt, -1);
      if (jt->timestamp - it->timestamp > 2500000)
        continue;
    }
    pairs.emplace_back(std::distance(trajectory.cbegin(), it),
                       std::distance(groundtruth.cbegin(), jt));
  }

  const int N = pairs.size();
  CHECK_GE(N, 3) << "At least 3 poses are required!";

  Eigen::MatrixXd src(3, N);
  Eigen::MatrixXd dst(3, N);

  int index = 0;
  for (const std::pair<unsigned, unsigned> &match : pairs) {
    const io::pose_t traj = trajectory.at(match.first).pose;
    const io::pose_t ref = groundtruth.at(match.second).pose;

    src.col(index) = Eigen::Vector3d(traj.tx, traj.ty, traj.tz);
    dst.col(index) = Eigen::Vector3d(ref.tx, ref.ty, ref.tz);
    index++;
  }

  Eigen::Matrix4d M = Eigen::umeyama(src, dst, true);

  scale_factor = std::cbrt(M.block<3, 3>(0, 0).determinant());

  Eigen::Isometry3d T;
  T.linear() = M.block<3, 3>(0, 0) / scale_factor;
  T.translation() = M.block<3, 1>(0, 3);
  return T;
}

void write_results_to_csv(const vector<method_result> &method_results, double true_scale) {

  for (auto mr : method_results) {
    fs::path filepath = "." / fs::path(mr.method_name + ".csv");
    vector<ResultType> results = mr.results;
    const int rows = results.size();
    const int cols = 12;
    Eigen::MatrixXd mat(rows, cols + 1);
    LOG(INFO) << "Writing results to disk: " << mr.method_name;
    for (int i = 0; i < rows; i++) {
      Eigen::RowVectorXd r(cols + 1);
      r << i, results[i].success, results[i].scale, true_scale,
          results[i].bias_g.transpose(), results[i].bias_g.transpose(),
          results[i].gravity.transpose();
      mat.row(i) = r;
    }
    string header = "# index, flag, s_computed, s_true, b_g_x, b_g_y, b_g_z, b_a_x, b_a_y, "
                    "b_a_z, g_x, g_y, g_z \n";
    csv::write(mat, filepath.string(), header);
  }
}
void write_result_to_txt_file(fs::path filepath, ResultType result) {
  fstream store_result;
  store_result.open(filepath.string(), ios::out | ios::app);
  store_result << "Proposed output solution with IMU alignment \n"
               << "scale = " << result.scale << endl
               << "bias_g = " << result.bias_g << endl
               << "bias_a = " << result.bias_a << endl
               << "gravity = " << result.gravity << endl
               << "-------------------------------------------" << endl;
  store_result.close();
}

void save(const std::vector<evaluation_t> &data, const std::string &save_path) {
  const int n = 7;

  Eigen::MatrixXd m(data.size(), n);

  for (unsigned i = 0; i < data.size(); ++i) {
    Eigen::RowVectorXd row(n);
    row << data[i].execution_time, data[i].scale_error, data[i].gyro_bias_error,
        data[i].gyro_bias_error2, data[i].acc_bias_error,
        data[i].acc_bias_error2, data[i].gravity_error;
    m.row(i) = row;
  }

  csv::write(m, save_path);
}

