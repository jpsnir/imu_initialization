
// 10 KFs at 4 Hz

#define PROGRAM_NAME "experiment01"

#define FLAGS_CASES                                                            \
  FLAG_CASE(string, logs_dir, "./logs/", "Logs save directory")

#define ARGS_CASES ARG_CASE(dataset_dir)

// STL
#include <algorithm>
#include <cmath>

#include <iterator>
#include <string>
#include <vector>

#include <stdexcept>

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

#include "data_utilities.h"
#include "util/args.h"
#include "util/csv.h"

namespace fs = boost::filesystem;

using Groundtruth = std::vector<io::state_t>;
using ImuData = io::ImuData;

// IMU parameters
// EuRoC
const double rate = 200.;
const double dt = 1. / rate;
const double ng = 1.7e-4;
const double na = 2.e-3;

struct evaluation_t {
  evaluation_t(const double scale_error, const double gyro_bias_error,
               const double gyro_bias_error2, const double acc_bias_error,
               double acc_bias_error2, const double gravity_error)
      : scale_error(scale_error), gyro_bias_error(gyro_bias_error),
        gyro_bias_error2(gyro_bias_error2), acc_bias_error(acc_bias_error),
        acc_bias_error2(acc_bias_error2), gravity_error(gravity_error) {}

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  double scale_error;      // percent
  double gyro_bias_error;  // percent
  double gyro_bias_error2; // degress
  double acc_bias_error;   // percent
  double acc_bias_error2;  // degrees
  double gravity_error;    // degrees
};

struct method_result {
  string method_name;
  vector<ResultType> results;
};

void ValidateArgs() { CHECK(fs::is_directory(ARGS_dataset_dir)); }

void ValidateFlags() { fs::create_directories(FLAGS_logs_dir); }

Groundtruth::const_iterator next(Groundtruth::const_iterator i,
                                 Groundtruth::const_iterator j,
                                 io::timestamp_t dt) {
  if (i == j) {
    LOG(WARNING) << "Already at the end...";
    return i;
  }

  io::timestamp_t t = i->timestamp + dt;
  Groundtruth::const_iterator it = std::lower_bound(
      i, j, t, [](const io::state_t &lhs, const io::timestamp_t rhs) {
        return lhs.timestamp < rhs;
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

Groundtruth::const_iterator start(const Groundtruth &trajectory,
                                  const io::ImuData &imu_data,
                                  const unsigned nframes) {
  Groundtruth::const_iterator i = trajectory.cbegin();
  Groundtruth::const_iterator i_ = i;
  while (i != trajectory.cend()) {
    Eigen::Vector3d avgA;
    avgA.setZero();

    io::ImuData::const_iterator it = imu_data.cbegin();
    for (unsigned n = 0; n < nframes; ++n) {
      it = start_imu(it, imu_data.cend(), i->timestamp);
      CHECK(it != imu_data.cend());

      Groundtruth::const_iterator j =
          next(i, trajectory.cend(), 250000000); // 4 Hz
      CHECK(j != trajectory.cend());

      std::shared_ptr<IMU::Preintegrated> pInt =
          std::make_shared<IMU::Preintegrated>(Eigen::Vector3d::Zero(),
                                               Eigen::Vector3d::Zero());
      while (it != imu_data.cend() &&
             std::llabs(it->timestamp - j->timestamp) > 2500000) {
        // it->timestamp < j->timestamp) {
        const Eigen::Vector3d w(it->w_x, it->w_y, it->w_z);
        const Eigen::Vector3d a(it->a_x, it->a_y, it->a_z);
        pInt->IntegrateNewMeasurement(w, a, dt);
        std::advance(it, 1);
      }
      CHECK(it != imu_data.cend());

      avgA += pInt->dV / pInt->dT;
      i = j;
    }

    avgA /= static_cast<double>(nframes);
    const double avgA_error =
        std::abs(avgA.norm() - IMU::GRAVITY_MAGNITUDE) / IMU::GRAVITY_MAGNITUDE;
    LOG(INFO) << "Average acceleration: " << 100. * avgA_error;
    if (avgA_error > 5e-3)
      break;

    i = next(i, trajectory.cend(), 500000000); // 0.5s
    i_ = i;
  }
  CHECK(i != trajectory.cend());

  return i_;
}

void save(const std::vector<evaluation_t> &data, const std::string &save_path) {

  const int N = data.size();
  const int n = 6;

  Eigen::MatrixXd m(N, n);

  for (int i = 0; i < N; ++i) {
    Eigen::RowVectorXd row(n);
    row << data[i].scale_error, data[i].gyro_bias_error,
        data[i].gyro_bias_error2, data[i].acc_bias_error,
        data[i].acc_bias_error2, data[i].gravity_error;
    m.row(i) = row;
  }

  csv::write(m, save_path);
}

void write_results_to_csv(const vector<method_result> &method_results) {

  for (auto mr : method_results) {
    fs::path filepath = "." / fs::path(mr.method_name + ".csv");
    vector<ResultType> results = mr.results;
    const int rows = results.size();
    const int cols = 11;
    Eigen::MatrixXd mat(rows, cols + 1);
    LOG(INFO) << "Writing results to disk: " << mr.method_name;
    for (int i = 0; i < rows; i++) {
      Eigen::RowVectorXd r(cols + 1);
      r << i, results[i].success, results[i].scale, results[i].bias_g.transpose(),
          results[i].bias_g.transpose(), results[i].gravity.transpose();
      mat.row(i) = r;
    }
    string header = "# index, flag, s, b_g_x, b_g_y, b_g_z, b_a_x, b_a_y, b_a_z, g_x, g_y, g_z \n";
    csv::write(mat, filepath.string(), header);
  }
}

void write_result_to_txt(fs::path filepath, ResultType result) {
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

void run(const fs::path &sequence_path) {

  std::string sequence_name = sequence_path.filename().string();
  if (sequence_name == ".")
    sequence_name = sequence_path.parent_path().filename().string();

  LOG(INFO) << "Running experiment: " << sequence_name;

  fs::path trajectory_path =
      sequence_path / "state_groundtruth_estimate0" / "data.csv";
  CHECK(fs::is_regular_file(trajectory_path))
      << "Path not found: " << trajectory_path.string();

  fs::path data_path = sequence_path / "imu0" / "data.csv";
  CHECK(fs::is_regular_file(data_path))
      << "Path not found: " << data_path.string();

  Groundtruth trajectory =
      io::read_file<Groundtruth::value_type>(trajectory_path.string());
  int n = trajectory.size();
  LOG(INFO) << "Ground truth trajectory start : " << trajectory[0].timestamp;
  LOG(INFO) << "Ground truth trajectory end :" << trajectory[n - 1].timestamp;
  LOG(INFO) << "delta time between gt trajectory "
            << trajectory[n - 1].timestamp - trajectory[0].timestamp;
  io::ImuData imu_data =
      io::read_file<io::ImuData::value_type>(data_path.string());
  int n1 = imu_data.size();
  LOG(INFO) << "imu_data  start :" << imu_data[0].timestamp
            << " gt - imu start "
            << trajectory[0].timestamp - imu_data[0].timestamp;
  LOG(INFO) << "imu_data  end :" << imu_data[n1 - 1].timestamp
            << " gt - imu end "
            << trajectory[n - 1].timestamp - imu_data[n1 - 1].timestamp;

  std::vector<unsigned> possible_nframes = {5, 10, 20, 50, 75};
  possible_nframes = slice_vector(possible_nframes, 0, 0);
  LOG(INFO) << "size of nframes: " << possible_nframes.size();
  int preintegration_counter = 0;
  for (unsigned nframes : possible_nframes) {
    Groundtruth::const_iterator i = start(trajectory, imu_data, nframes);
    LOG(INFO) << "Starting at " << i->timestamp;
    LOG(INFO) << StringPrintf("With %d frames", nframes);

    // saves the evaluation for each n frame size.
    std::vector<evaluation_t> proposed_evaluation;
    std::vector<evaluation_t> proposed_noprior_evaluation;
    std::vector<evaluation_t> iterative_evaluation;
    std::vector<evaluation_t> iterative_noprior_evaluation;
    std::vector<evaluation_t> mqh_evaluation;

    // method results
    std::vector<method_result> methods(5);

    unsigned count = 0;
    // std::uint64_t imu_integration = 0;

    std::uint64_t skipped = 0;
    Groundtruth::const_iterator i_ = i;

    // build the dataset for defining initialization problem for the complete
    // trajectory. Here we are trying to extract segments of data to solve the
    // initialization problem the input_t struct defines the data input.
    while (i != trajectory.cend()) {
      InputType input;

      Eigen::Vector3d avgBg = Eigen::Vector3d(i->bw_x, i->bw_y, i->bw_z);
      Eigen::Vector3d avgBa = Eigen::Vector3d(i->ba_x, i->ba_y, i->ba_z);
      Eigen::Vector3d avgA;
      avgA.setZero();
      LOG_EVERY_N(INFO, 500) << "Biases :  a = " << avgBa << " g = " << avgBg;
      io::ImuData::const_iterator it = imu_data.cbegin();

      // Collected IMU data for n frames.
      for (unsigned n = 0; n < nframes; ++n) {
        it = start_imu(it, imu_data.cend(), i->timestamp);
        if (it == imu_data.cend()) {
          LOG(WARNING) << "Couldn't find IMU measurement at " << i->timestamp;
          break;
        }
        LOG(INFO) << "Imu time stamp begin: " << it->timestamp
                  << " gt timestamp : " << i->timestamp
                  << " diff : " << i->timestamp - it->timestamp;

        Groundtruth::const_iterator j =
            next(i, trajectory.cend(), 250000000); // 4 Hz
        LOG_EVERY_N(INFO, 10)
            << "ground truth trajectory timestamp : " << j->timestamp;
        if (j == trajectory.cend()) {
          LOG(WARNING) << "Couldn't find next frame for " << i->timestamp;
          break;
        }

        // Timer timer;
        // timer.Start();

        LOG(INFO) << "Preintegration started: " << preintegration_counter++;
        std::shared_ptr<IMU::Preintegrated> pInt =
            std::make_shared<IMU::Preintegrated>(Eigen::Vector3d::Zero(),
                                                 Eigen::Vector3d::Zero());
        while (it != imu_data.cend() &&
               std::llabs(it->timestamp - j->timestamp) > 2500000) {
          // it->timestamp < j->timestamp) {
          const Eigen::Vector3d w(it->w_x, it->w_y, it->w_z);
          const Eigen::Vector3d a(it->a_x, it->a_y, it->a_z);
          pInt->IntegrateNewMeasurement(w, a, dt);
          std::advance(it, 1);
        }

        if (it == imu_data.cend()) {
          LOG(WARNING) << "IMU stream ended!";
          break;
        }

        // imu_integration += timer.ElapsedNanoSeconds();
        count++;

        avgBg += Eigen::Vector3d(j->bw_x, j->bw_y, j->bw_z);
        avgBa += Eigen::Vector3d(j->ba_x, j->ba_y, j->ba_z);

        avgA += pInt->dV / pInt->dT;
        input.emplace_back(i->pose, i->timestamp, j->pose, j->timestamp, pInt);

        // update the iterator for ground truth trajectory
        i = j;
      }

      if (input.size() < nframes) {
        LOG(INFO) << StringPrintf(
            "I don't have %d frames. I think dataset ended...", nframes);
        break;
      }

      avgBg /= static_cast<double>(nframes + 1);
      avgBa /= static_cast<double>(nframes + 1);

      avgA /= static_cast<double>(nframes);
      const double avgA_error = std::abs(avgA.norm() - IMU::GRAVITY_MAGNITUDE) /
                                IMU::GRAVITY_MAGNITUDE;

      LOG(INFO) << "Input for the problem - size : " << input.size();
      LOG(INFO) << "Average acceleration error : " << avgA_error;
      LOG(INFO) << " Calibration started  ";
      char c;
      // cin >> c;
      //  Compute average acceleration error to know if the calibration is
      //  needed. THis is trying to compute the error based on average
      //  acceleration in the preintegration time frame If the magnitude is
      //  nearly 9,81 we can assume that there is no acceleration the system is
      //  already calibrated
      if (avgA_error > 5e-3) {
        std::uint64_t timestamp = input[0].t1;
        // std::uint64_t initialization_time = i->timestamp - i_->timestamp;

        // Method 1: Proposed method with exact solution from the roots of the
        // polynomial.
        {
          ResultType gyroscope_result;
          gyroscope_only(input, gyroscope_result);
          LOG(INFO) << "Gyroscopic calibration ended : ";
          LOG(INFO) << "gyro bias success:  " << gyroscope_result.success;
          LOG(INFO) << "gyro bias: \n " << gyroscope_result.bias_g;

          ResultType accelerometer_result;
          analytic_accelerometer(
              input, accelerometer_result, gyroscope_result.bias_g,
              Eigen::Vector3d::Zero(), Eigen::Isometry3d::Identity(), 1e5);
          LOG(INFO) << " acc bias calibration ended " << endl;
          LOG(INFO) << " acc bias success : " << accelerometer_result.success;
          LOG(INFO) << " acc bias : \n" << accelerometer_result.bias_a;

          ResultType proposed_result;
          proposed_result.success =
              gyroscope_result.success & accelerometer_result.success;
          proposed_result.solve_ns =
              gyroscope_result.solve_ns + accelerometer_result.solve_ns;
          proposed_result.scale = accelerometer_result.scale;
          proposed_result.bias_g = gyroscope_result.bias_g;
          proposed_result.bias_a = accelerometer_result.bias_a;
          proposed_result.gravity = accelerometer_result.gravity;
          if (proposed_result.success) {
            const double scale_error =
                100. * std::abs(proposed_result.scale - 1.);
            const double gyro_bias_error =
                100. * std::abs(proposed_result.bias_g.norm() - avgBg.norm()) /
                avgBg.norm();
            const double gyro_bias_error2 =
                180. *
                std::acos(proposed_result.bias_g.normalized().dot(
                    avgBg.normalized())) /
                EIGEN_PI;
            const double acc_bias_error =
                100. * std::abs(proposed_result.bias_a.norm() - avgBa.norm()) /
                avgBa.norm();
            const double acc_bias_error2 =
                180. *
                std::acos(proposed_result.bias_a.normalized().dot(
                    avgBa.normalized())) /
                EIGEN_PI;
            const double gravity_error =
                180. *
                std::acos(proposed_result.gravity.normalized().dot(
                    IMU::GRAVITY_VECTOR.normalized())) /
                EIGEN_PI;
            proposed_evaluation.emplace_back(scale_error, gyro_bias_error,
                                             gyro_bias_error2, acc_bias_error,
                                             acc_bias_error2, gravity_error);
          } else
            LOG(ERROR) << "Proposed method failed at " << timestamp;
          methods[0].method_name = "proposed";
          methods[0].results.push_back(proposed_result);
          write_result_to_txt("./proposed_result.txt", proposed_result);
        }

        // method 2: proposed method without giving a prior
        {
          ResultType gyroscope_result;
          gyroscope_only(input, gyroscope_result);

          ResultType accelerometer_result;
          analytic_accelerometer(
              input, accelerometer_result, gyroscope_result.bias_g,
              Eigen::Vector3d::Zero(), Eigen::Isometry3d::Identity(), 0.);

          ResultType proposed_result;
          proposed_result.success =
              gyroscope_result.success & accelerometer_result.success;
          proposed_result.solve_ns =
              gyroscope_result.solve_ns + accelerometer_result.solve_ns;
          proposed_result.scale = accelerometer_result.scale;
          proposed_result.bias_g = gyroscope_result.bias_g;
          proposed_result.bias_a = accelerometer_result.bias_a;
          proposed_result.gravity = accelerometer_result.gravity;

          if (proposed_result.success) {
            const double scale_error =
                100. * std::abs(proposed_result.scale - 1.);
            const double gyro_bias_error =
                100. * std::abs(proposed_result.bias_g.norm() - avgBg.norm()) /
                avgBg.norm();
            const double gyro_bias_error2 =
                180. *
                std::acos(proposed_result.bias_g.normalized().dot(
                    avgBg.normalized())) /
                EIGEN_PI;
            const double acc_bias_error =
                100. * std::abs(proposed_result.bias_a.norm() - avgBa.norm()) /
                avgBa.norm();
            const double acc_bias_error2 =
                180. *
                std::acos(proposed_result.bias_a.normalized().dot(
                    avgBa.normalized())) /
                EIGEN_PI;
            const double gravity_error =
                180. *
                std::acos(proposed_result.gravity.normalized().dot(
                    IMU::GRAVITY_VECTOR.normalized())) /
                EIGEN_PI;
            proposed_noprior_evaluation.emplace_back(
                scale_error, gyro_bias_error, gyro_bias_error2, acc_bias_error,
                acc_bias_error2, gravity_error);
          } else
            LOG(ERROR) << "Proposed w/o prior method failed at " << timestamp;

          methods[1].method_name = "proposed_wo_prior";
          methods[1].results.push_back(proposed_result);

          write_result_to_txt("./result_proposed_prior.txt", proposed_result);
        }

        {
          ResultType iterative_result;
          iterative(input, iterative_result, 1., Eigen::Isometry3d::Identity(),
                    nullptr, 1e5);

          if (iterative_result.success) {
            const double scale_error =
                100. * std::abs(iterative_result.scale - 1.);
            const double gyro_bias_error =
                100. * std::abs(iterative_result.bias_g.norm() - avgBg.norm()) /
                avgBg.norm();
            const double gyro_bias_error2 =
                180. *
                std::acos(iterative_result.bias_g.normalized().dot(
                    avgBg.normalized())) /
                EIGEN_PI;
            const double acc_bias_error =
                100. * std::abs(iterative_result.bias_a.norm() - avgBa.norm()) /
                avgBa.norm();
            const double acc_bias_error2 =
                180. *
                std::acos(iterative_result.bias_a.normalized().dot(
                    avgBa.normalized())) /
                EIGEN_PI;
            const double gravity_error =
                180. *
                std::acos(iterative_result.gravity.normalized().dot(
                    IMU::GRAVITY_VECTOR.normalized())) /
                EIGEN_PI;
            iterative_evaluation.emplace_back(scale_error, gyro_bias_error,
                                              gyro_bias_error2, acc_bias_error,
                                              acc_bias_error2, gravity_error);
          } else
            LOG(ERROR) << "Iterative method failed at " << timestamp;
          methods[2].method_name = "iterative";
          methods[2].results.push_back(iterative_result);
          write_result_to_txt("./result_iterative.txt", iterative_result);
        }

        {
          ResultType iterative_result;
          iterative(input, iterative_result, 1., Eigen::Isometry3d::Identity(),
                    nullptr, 0.);

          if (iterative_result.success) {
            const double scale_error =
                100. * std::abs(iterative_result.scale - 1.);
            const double gyro_bias_error =
                100. * std::abs(iterative_result.bias_g.norm() - avgBg.norm()) /
                avgBg.norm();
            const double gyro_bias_error2 =
                180. *
                std::acos(iterative_result.bias_g.normalized().dot(
                    avgBg.normalized())) /
                EIGEN_PI;
            const double acc_bias_error =
                100. * std::abs(iterative_result.bias_a.norm() - avgBa.norm()) /
                avgBa.norm();
            const double acc_bias_error2 =
                180. *
                std::acos(iterative_result.bias_a.normalized().dot(
                    avgBa.normalized())) /
                EIGEN_PI;
            const double gravity_error =
                180. *
                std::acos(iterative_result.gravity.normalized().dot(
                    IMU::GRAVITY_VECTOR.normalized())) /
                EIGEN_PI;
            iterative_noprior_evaluation.emplace_back(
                scale_error, gyro_bias_error, gyro_bias_error2, acc_bias_error,
                acc_bias_error2, gravity_error);
          } else
            LOG(ERROR) << "Iterative w/o prior method failed at " << timestamp;
          methods[3].method_name = "iterative_wo_prior";
          methods[3].results.push_back(iterative_result);
          write_result_to_txt("./result_iterative_prior.txt", iterative_result);
        }

        {
          ResultType gyroscope_result;
          gyroscope_only(input, gyroscope_result, Eigen::Matrix3d::Identity(),
                         false);

          ResultType accelerometer_result;
          mqh_accelerometer(input, accelerometer_result,
                            gyroscope_result.bias_g,
                            Eigen::Isometry3d::Identity());

          ResultType mqh_result;
          mqh_result.success =
              gyroscope_result.success & accelerometer_result.success;
          // proposed_result.solve_ns = gyroscope_result.solve_ns +
          // accelerometer_result.solve_ns;
          mqh_result.scale = accelerometer_result.scale;
          mqh_result.bias_g = gyroscope_result.bias_g;
          mqh_result.bias_a = accelerometer_result.bias_a;
          mqh_result.gravity = accelerometer_result.gravity;

          if (mqh_result.success) {
            const double scale_error = 100. * std::abs(mqh_result.scale - 1.);
            const double gyro_bias_error =
                100. * std::abs(mqh_result.bias_g.norm() - avgBg.norm()) /
                avgBg.norm();
            const double gyro_bias_error2 =
                180. *
                std::acos(
                    mqh_result.bias_g.normalized().dot(avgBg.normalized())) /
                EIGEN_PI;
            const double acc_bias_error =
                100. * std::abs(mqh_result.bias_a.norm() - avgBa.norm()) /
                avgBa.norm();
            const double acc_bias_error2 =
                180. *
                std::acos(
                    mqh_result.bias_a.normalized().dot(avgBa.normalized())) /
                EIGEN_PI;
            const double gravity_error =
                180. *
                std::acos(mqh_result.gravity.normalized().dot(
                    IMU::GRAVITY_VECTOR.normalized())) /
                EIGEN_PI;
            mqh_evaluation.emplace_back(scale_error, gyro_bias_error,
                                        gyro_bias_error2, acc_bias_error,
                                        acc_bias_error2, gravity_error);
          } else
            LOG(ERROR) << "MQH method failed at " << timestamp;
          methods[4].method_name = "mqh";
          methods[4].results.push_back(mqh_result);
          write_result_to_txt("./result_mqh.txt", mqh_result);
        }

        i = next(i_, trajectory.cend(), 500000000);
        i_ = i;
        skipped = 0;
      } else { // next attempt
        skipped += 500000000;
        i = next(i_, trajectory.cend(), skipped); // 0.5s
      }
    }

    std::string proposed_file =
        StringPrintf("%s_%d_ours.csv", sequence_name.c_str(), nframes);
    LOG(INFO) << "Saving evaluation data into " << proposed_file;
    save(proposed_evaluation, proposed_file);

    std::string proposed_noprior_file =
        StringPrintf("%s_%d_ours_noprior.csv", sequence_name.c_str(), nframes);
    LOG(INFO) << "Saving evaluation data into " << proposed_noprior_file;
    save(proposed_noprior_evaluation, proposed_noprior_file);

    std::string iterative_file =
        StringPrintf("%s_%d_iterative.csv", sequence_name.c_str(), nframes);
    LOG(INFO) << "Saving evaluation data into " << iterative_file;
    save(iterative_evaluation, iterative_file);

    std::string iterative_noprior_file = StringPrintf(
        "%s_%d_iterative_noprior.csv", sequence_name.c_str(), nframes);
    LOG(INFO) << "Saving evaluation data into " << iterative_noprior_file;
    save(iterative_noprior_evaluation, iterative_noprior_file);

    std::string mqh_file =
        StringPrintf("%s_%d_mqh.csv", sequence_name.c_str(), nframes);
    LOG(INFO) << "Saving evaluation data into " << mqh_file;
    save(mqh_evaluation, mqh_file);

    // saving results
    write_results_to_csv(methods);
  }

  LOG(INFO) << "done." << std::endl;
}

int main(int argc, char *argv[]) {

  // Handle help flag
  if (args::HelpRequired(argc, argv)) {
    args::ShowHelp();
    return 0;
  }

  // Parse input flags
  args::ParseCommandLineNonHelpFlags(&argc, &argv, true);

  FLAGS_log_dir = FLAGS_logs_dir;
  FLAGS_stderrthreshold = 0;
  google::InitGoogleLogging(argv[0]);

  // Check number of args
  if (argc - 1 != args::NumArgs()) {
    args::ShowHelp();
    return -1;
  }

  // Parse input args
  args::ParseCommandLineArgs(argc, argv);

  // Validate input arguments
  ValidateFlags();
  ValidateArgs();

  IMU::Sigma.block<3, 3>(0, 0) = rate * ng * ng * Eigen::Matrix3d::Identity();
  IMU::Sigma.block<3, 3>(3, 3) = rate * na * na * Eigen::Matrix3d::Identity();

  run(ARGS_dataset_dir);

  return 0;
}
