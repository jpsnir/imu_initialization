
#include "experiment02.h"
#include <unistd.h>
DEFINE_uint32(camera_step_size, 1, "camera step size in the estimated keyframe trajectory");
DEFINE_double(gap_time, 0.5, "gap time for search keyframes from current timestamp - seconds");
DEFINE_string(filename, "KeyFrameTrajectory.txt", "camera trajectory file obtained from VO algorithm");

void run(const fs::path &sequence_path)
{
  std::string sequence_name = sequence_path.filename().string();
  if (sequence_name == ".")
    sequence_name = sequence_path.parent_path().filename().string();

  LOG(INFO) << "Running experiment: " << sequence_name;
  LOG(INFO) << StringPrintf("With %d keyframes", FLAGS_nframes);
  fs::path trajectory_path = sequence_path / FLAGS_filename;
  unsigned pos = FLAGS_filename.find_last_of(".");
  string dataset_name = FLAGS_filename.substr(0, pos);
  CHECK(fs::is_regular_file(trajectory_path)) << "Path not found: " << trajectory_path.string();


  fs::path groundtruth_path = sequence_path / "state_groundtruth_estimate0" / "data.csv";
  CHECK(fs::is_regular_file(groundtruth_path)) << "Path not found: " << groundtruth_path.string();

  fs::path data_path = sequence_path / "imu0" / "data.csv";
  CHECK(fs::is_regular_file(data_path)) << "Path not found: " << data_path.string();

  Trajectory trajectory_ = read_file_TUM(trajectory_path.string());
  Groundtruth groundtruth = io::read_file<Groundtruth::value_type>(groundtruth_path.string());
  io::ImuData imu_data = io::read_file<io::ImuData::value_type>(data_path.string());

  // Discard first keyframe
  Trajectory trajectory(std::next(trajectory_.cbegin(), gap_time), trajectory_.cend());
  Trajectory::const_iterator i = start(trajectory, imu_data);
  LOG(INFO) << "Starting at " << static_cast<io::timestamp_t>(i->timestamp * 1e9);
  std::vector<evaluation_t> proposed_evaluation;
  std::vector<evaluation_t> proposed_noprior_evaluation;
  std::vector<evaluation_t> iterative_evaluation;
  std::vector<evaluation_t> iterative_noprior_evaluation;
  std::vector<evaluation_t> mqh_evaluation;
  std::vector<method_result> methods(5);
  double true_scale = std::numeric_limits<double>::quiet_NaN();

  double skipped = 0.;
  unsigned iter = 0;
  Trajectory::const_iterator i_ = i;
  cout << std::fixed << std::setprecision(10);

  /* MAIN LOOP */
  while (i != trajectory.cend())
  {
    LOG(INFO) << "******** ITERATION ID: " << iter++ << "**************";
    Groundtruth::const_iterator gt = find_closest(groundtruth.cbegin(), groundtruth.cend(),
                                                  static_cast<io::timestamp_t>(i->timestamp * 1e9));
    if (gt == groundtruth.cend())
    {
      LOG(WARNING) << "Couldn't find groundtruth for " << static_cast<io::timestamp_t>(i->timestamp * 1e9);
      LOG(INFO) << "breaking at: " << i->timestamp * 1e9 << ":" << i_->timestamp * 1e9;
      break;
    }
    CHECK(gt != groundtruth.cend());
    InputType input;
    Eigen::Vector3d avgBg = Eigen::Vector3d(gt->bw_x, gt->bw_y, gt->bw_z);
    Eigen::Vector3d avgBa = Eigen::Vector3d(gt->ba_x, gt->ba_y, gt->ba_z);
    Eigen::Vector3d avgA;
    avgA.setZero();
    std::uint64_t imu_preintegration = 0;
    io::ImuData::const_iterator it = imu_data.cbegin();

    // define the input with camera poses and IMU readings.
    for (unsigned n = 0; n < FLAGS_nframes; ++n)
    {
      it = start_imu(it, imu_data.cend(), static_cast<io::timestamp_t>(i->timestamp * 1e9));
      if (it == imu_data.cend())
      {
        LOG(WARNING) << "Couldn't find IMU measurement at " << static_cast<io::timestamp_t>(i->timestamp * 1e9);
        break;
      }
      Trajectory::const_iterator j = std::next(i, c_step_size);
      LOG(INFO) << " Forming input: Timestamp of current keyframe : " << fixed << setprecision(10) << i->timestamp;
      LOG(INFO) << " Forming input: Timestamp of next keyframe : " << fixed << setprecision(10) << j->timestamp;
      if (j == trajectory.cend())
      {
        LOG(WARNING) << "Couldn't find next frame for "
                     << static_cast<io::timestamp_t>(i->timestamp * 1e9);
        break;
      }
      gt = find_closest(groundtruth.cbegin(), groundtruth.cend(), static_cast<io::timestamp_t>(j->timestamp * 1e9));
      if (gt == groundtruth.cend())
      {
        LOG(WARNING) << "Couldn't find groundtruth for "
                     << static_cast<io::timestamp_t>(j->timestamp * 1e9);
        break;
      }
      avgBg += Eigen::Vector3d(gt->bw_x, gt->bw_y, gt->bw_z);
      avgBa += Eigen::Vector3d(gt->ba_x, gt->ba_y, gt->ba_z);

      Timer timer;
      timer.Start();

      std::shared_ptr<IMU::Preintegrated> pInt =
          std::make_shared<IMU::Preintegrated>(Eigen::Vector3d::Zero(),
                                               Eigen::Vector3d::Zero());
      while (it != imu_data.cend() &&
             std::abs(it->timestamp * 1e-9 - j->timestamp) > 0.0025)
      {
        const Eigen::Vector3d w(it->w_x, it->w_y, it->w_z);
        const Eigen::Vector3d a(it->a_x, it->a_y, it->a_z);
        pInt->IntegrateNewMeasurement(w, a, dt);
        std::advance(it, 1);
      }
      imu_preintegration += timer.ElapsedNanoSeconds();

      if (it == imu_data.cend())
      {
        LOG(WARNING) << "IMU stream ended!";
        break;
      }

      avgA += pInt->dV / pInt->dT;
      input.emplace_back(i->pose,
                         static_cast<io::timestamp_t>(i->timestamp * 1e9),
                         j->pose,
                         static_cast<io::timestamp_t>(j->timestamp * 1e9),
                         pInt);
      i = j;
    }

    LOG(INFO) << "Time difference = " << i->timestamp - i_->timestamp << " : counter = " << iter;

    // check defined input
    if (input.size() < FLAGS_nframes)
    {
      LOG(INFO) << "I don't have " << FLAGS_nframes
                << " frames. I think dataset ended...";
      break;
    }
    avgBg /= static_cast<double>(FLAGS_nframes + 1);
    avgBa /= static_cast<double>(FLAGS_nframes + 1);
    avgA /= static_cast<double>(FLAGS_nframes);

    const double avgA_error = std::abs(avgA.norm() - IMU::GRAVITY_MAGNITUDE) / IMU::GRAVITY_MAGNITUDE;
    LOG(INFO) << "Average acceleration: " << avgA_error;
    if (avgA_error > 5e-3)
    {
      std::uint64_t timestamp = input[0].t1;
 
      Eigen::Isometry3d T = compute_scale(input, groundtruth, true_scale);

      // if (true_scale == -1){
      //   LOG(INFO) << "True scale = -1. Rebuilding input.";
      //   continue;
      // }
        

      // Method 1: Proposed solution
      {
        ResultType gyroscope_result;
        gyroscope_only(input,
                       gyroscope_result,
                       Tcb.linear());

        ResultType accelerometer_result;
        analytic_accelerometer(input,
                               accelerometer_result,
                               gyroscope_result.bias_g,
                               Eigen::Vector3d::Zero(),
                               Tcb, 1e5);

        ResultType proposed_result;
        proposed_result.success = gyroscope_result.success & accelerometer_result.success;
        proposed_result.solve_ns = gyroscope_result.solve_ns +
                                   accelerometer_result.solve_ns +
                                   accelerometer_result.velocities_ns;
        proposed_result.scale = accelerometer_result.scale;
        proposed_result.bias_g = gyroscope_result.bias_g;
        proposed_result.bias_a = accelerometer_result.bias_a;
        proposed_result.gravity = accelerometer_result.gravity;

        if (proposed_result.success)
        {
          const double scale_error = 100. * std::abs(proposed_result.scale - true_scale) / true_scale;
          const double gyro_bias_error = 100. * std::abs(proposed_result.bias_g.norm() - avgBg.norm()) /
                                         avgBg.norm();
          const double gyro_bias_error2 = 180. *
                                          std::acos(
                                              proposed_result.bias_g.normalized().dot(avgBg.normalized())) /
                                          EIGEN_PI;
          const double acc_bias_error = 100. * std::abs(proposed_result.bias_a.norm() - avgBa.norm()) /
                                        avgBa.norm();
          const double acc_bias_error2 = 180. *
                                         std::acos(
                                             proposed_result.bias_a.normalized().dot(avgBa.normalized())) /
                                         EIGEN_PI;
          const double gravity_error = 180. *
                                       std::acos(
                                           (T.linear() * proposed_result.gravity).normalized().dot(IMU::GRAVITY_VECTOR.normalized())) /
                                       EIGEN_PI;
          proposed_evaluation.emplace_back(imu_preintegration + proposed_result.solve_ns, 
                                           scale_error, gyro_bias_error, gyro_bias_error2, acc_bias_error,
                                           acc_bias_error2, gravity_error);
        }
        else
          LOG(ERROR) << "Proposed method failed at " << timestamp;

        methods[0].dataset_name = dataset_name;
        methods[0].method_name = "proposed_cam" + std::to_string(FLAGS_nframes);
        methods[0].results.push_back(proposed_result);
        // write_result_to_txt_file("./proposed_camera_result.txt",
        //                          proposed_result);
        LOG(INFO) << " Method 1 completed";
      }

      // Method 2: Proposed without prior
      {
        ResultType gyroscope_result;
        gyroscope_only(input, gyroscope_result, Tcb.linear());

        ResultType accelerometer_result;
        analytic_accelerometer(input, accelerometer_result,
                               gyroscope_result.bias_g, Eigen::Vector3d::Zero(),
                               Tcb, 0.0);

        ResultType proposed_result;
        proposed_result.success =
            gyroscope_result.success & accelerometer_result.success;
        proposed_result.solve_ns = gyroscope_result.solve_ns +
                                   accelerometer_result.solve_ns +
                                   accelerometer_result.velocities_ns;
        proposed_result.scale = accelerometer_result.scale;
        proposed_result.bias_g = gyroscope_result.bias_g;
        proposed_result.bias_a = accelerometer_result.bias_a;
        proposed_result.gravity = accelerometer_result.gravity;

        if (proposed_result.success)
        {
          const double scale_error = 100. * std::abs(proposed_result.scale - true_scale) / true_scale;
          const double gyro_bias_error = 100. * std::abs(proposed_result.bias_g.norm() - avgBg.norm()) /
                                         avgBg.norm();
          const double gyro_bias_error2 = 180. *
                                          std::acos(
                                              proposed_result.bias_g.normalized().dot(avgBg.normalized())) /
                                          EIGEN_PI;
          const double acc_bias_error = 100. * std::abs(proposed_result.bias_a.norm() - avgBa.norm()) /
                                        avgBa.norm();
          const double acc_bias_error2 = 180. *
                                         std::acos(
                                             proposed_result.bias_a.normalized().dot(avgBa.normalized())) /
                                         EIGEN_PI;
          const double gravity_error = 180. *
                                       std::acos((T.linear() * proposed_result.gravity)
                                                     .normalized()
                                                     .dot(IMU::GRAVITY_VECTOR.normalized())) /
                                       EIGEN_PI;
          proposed_noprior_evaluation.emplace_back(imu_preintegration + proposed_result.solve_ns, scale_error,
                                                   gyro_bias_error, gyro_bias_error2, acc_bias_error,
                                                   acc_bias_error2, gravity_error);
        }
        else
          LOG(ERROR) << "Proposed w/o prior method failed at " << timestamp;
        methods[1].dataset_name = dataset_name;
        methods[1].method_name = "proposed_wo_prior_cam" + std::to_string(FLAGS_nframes);
        methods[1].results.push_back(proposed_result);
        // write_result_to_txt_file("./proposed_camera_result_wo_prior.txt",
        //                          proposed_result);
        LOG(INFO) << " Method 2 completed";
      }

      // Method 3: Iterative
      {
        ResultType iterative_result;
        double min_cost = std::numeric_limits<double>::max();

        std::int64_t max_solve_time = 0;
        std::vector<double> scale_values = {1., 4., 16.};
        for (const double scale : scale_values)
        {
          double cost;
          ResultType result;
          // LOG(INFO) << "Initializing with scale " << scale;
          iterative(input, result, scale, Tcb, &cost);
          max_solve_time = std::max(result.solve_ns, max_solve_time);
          if (cost < min_cost)
          {
            iterative_result = result;
            min_cost = cost;
          }
        }
        iterative_result.solve_ns = max_solve_time;

        if (iterative_result.success)
        {
          const double scale_error = 100. * std::abs(iterative_result.scale - true_scale) / true_scale;
          const double gyro_bias_error = 100. * std::abs(iterative_result.bias_g.norm() - avgBg.norm()) /
                                         avgBg.norm();
          const double gyro_bias_error2 = 180. *
                                          std::acos(iterative_result.bias_g.normalized().dot(
                                              avgBg.normalized())) /
                                          EIGEN_PI;
          const double acc_bias_error = 100. * std::abs(iterative_result.bias_a.norm() - avgBa.norm()) /
                                        avgBa.norm();
          const double acc_bias_error2 = 180. *
                                         std::acos(iterative_result.bias_a.normalized().dot(
                                             avgBa.normalized())) /
                                         EIGEN_PI;
          const double gravity_error = 180. *
                                       std::acos((T.linear() * iterative_result.gravity)
                                                     .normalized()
                                                     .dot(IMU::GRAVITY_VECTOR.normalized())) /
                                       EIGEN_PI;
          iterative_evaluation.emplace_back(imu_preintegration + iterative_result.solve_ns, scale_error,
                                            gyro_bias_error, gyro_bias_error2, acc_bias_error,
                                            acc_bias_error2, gravity_error);
        }
        else
          LOG(ERROR) << "Iterative method failed at " << timestamp;
        methods[2].dataset_name = dataset_name;
        methods[2].method_name = "iterative_cam" + std::to_string(FLAGS_nframes);
        methods[2].results.push_back(iterative_result);

        // write_result_to_txt_file("./iterative_camera_result.txt",
        //                          iterative_result);
      }

      // Method 4: Iterative without prior
      {
        ResultType iterative_result;
        double min_cost = std::numeric_limits<double>::max();

        std::int64_t max_solve_time = 0;
        std::vector<double> scale_values = {1., 4., 16.};
        for (const double scale : scale_values)
        {
          double cost;
          ResultType result;
          // LOG(INFO) << "Initializing with scale " << scale;
          iterative(input, result, scale, Tcb, &cost, 0.0);
          max_solve_time = std::max(result.solve_ns, max_solve_time);
          if (cost < min_cost)
          {
            iterative_result = result;
            min_cost = cost;
          }
        }
        iterative_result.solve_ns = max_solve_time;

        if (iterative_result.success)
        {
          const double scale_error = 100. * std::abs(iterative_result.scale - true_scale) / true_scale;
          const double gyro_bias_error =
              100. * std::abs(iterative_result.bias_g.norm() - avgBg.norm()) /
              avgBg.norm();
          const double gyro_bias_error2 = 180. *
                                          std::acos(iterative_result.bias_g.normalized().dot(
                                              avgBg.normalized())) /
                                          EIGEN_PI;
          const double acc_bias_error = 100. * std::abs(iterative_result.bias_a.norm() - avgBa.norm()) /
                                        avgBa.norm();
          const double acc_bias_error2 = 180. *
                                         std::acos(iterative_result.bias_a.normalized().dot(
                                             avgBa.normalized())) /
                                         EIGEN_PI;
          const double gravity_error = 180. *
                                       std::acos((T.linear() * iterative_result.gravity)
                                                     .normalized()
                                                     .dot(IMU::GRAVITY_VECTOR.normalized())) /
                                       EIGEN_PI;
          iterative_noprior_evaluation.emplace_back(imu_preintegration + iterative_result.solve_ns, scale_error,
                                                    gyro_bias_error, gyro_bias_error2, acc_bias_error,
                                                    acc_bias_error2, gravity_error);
        }
        else
          LOG(ERROR) << "Iterative w/o prior method failed at " << timestamp;
        methods[3].dataset_name = dataset_name;
        methods[3].method_name = "iterative_wo_prior_cam" + std::to_string(FLAGS_nframes);
        methods[3].results.push_back(iterative_result);

        // write_result_to_txt_file("./iterative_camera_wo_prior.txt",
        //                          iterative_result);
        LOG(INFO) << " Method 3 completed";
      }

      // Method 5: MQH
      {
        ResultType gyroscope_result;
        gyroscope_only(input, gyroscope_result, Tcb.linear(), false);

        ResultType accelerometer_result;
        mqh_accelerometer(input, accelerometer_result, gyroscope_result.bias_g,
                          Tcb);

        ResultType mqh_result;
        mqh_result.success =
            gyroscope_result.success & accelerometer_result.success;
        mqh_result.solve_ns = gyroscope_result.solve_ns +
                              accelerometer_result.solve_ns +
                              accelerometer_result.velocities_ns;
        mqh_result.scale = accelerometer_result.scale;
        mqh_result.bias_g = gyroscope_result.bias_g;
        mqh_result.bias_a = accelerometer_result.bias_a;
        mqh_result.gravity = accelerometer_result.gravity;

        if (mqh_result.success)
        {
          const double scale_error = 100. * std::abs(mqh_result.scale - 1.);
          const double gyro_bias_error = 100. * std::abs(mqh_result.bias_g.norm() - avgBg.norm()) /
                                         avgBg.norm();
          const double gyro_bias_error2 = 180. *
                                          std::acos(
                                              mqh_result.bias_g.normalized().dot(avgBg.normalized())) /
                                          EIGEN_PI;
          const double acc_bias_error = 100. * std::abs(mqh_result.bias_a.norm() - avgBa.norm()) /
                                        avgBa.norm();
          const double acc_bias_error2 = 180. *
                                         std::acos(
                                             mqh_result.bias_a.normalized().dot(avgBa.normalized())) /
                                         EIGEN_PI;
          const double gravity_error = 180. *
                                       std::acos(mqh_result.gravity.normalized().dot(
                                           IMU::GRAVITY_VECTOR.normalized())) /
                                       EIGEN_PI;
          mqh_evaluation.emplace_back(imu_preintegration + mqh_result.solve_ns,
                                      scale_error, gyro_bias_error,
                                      gyro_bias_error2, acc_bias_error,
                                      acc_bias_error2, gravity_error);
        }
        else
          LOG(ERROR) << "MQH method failed at " << timestamp;
        methods[4].dataset_name = dataset_name;
        methods[4].method_name = "mqh_camera" + std::to_string(FLAGS_nframes);
        methods[4].results.push_back(mqh_result);

        // write_result_to_txt_file("./mqh_result.txt", mqh_result);
        LOG(INFO) << " Method 4 completed";
      }

      i = next(i_, trajectory.cend(), gap_time);
      // TODO
      if (i->timestamp == i_->timestamp){
        LOG(INFO) << " No new frames are being selected in skip time = " << gap_time;
        LOG(INFO) << " Current keyframe timestamp: " << std::fixed << std::setprecision(10) << i_->timestamp;
        i = std::next(i, c_step_size);
        LOG(INFO) << " Jumping to next keyframe : " << std::fixed << std::setprecision(10) << i->timestamp;
        break;
      }
      else{
        i_ = i;
      }
      skipped = 0.;
    }
    else
    { // next attempt
      LOG(INFO) << "Average acceleration error nearly zero. Skipping frame.";
      skipped += gap_time;
      i = next(i_, trajectory.cend(), skipped); // 0.5s
    }
  }

  std::string proposed_file = sequence_name + "_ours.csv";
  LOG(INFO) << "Saving evaluation data into " << proposed_file;
  save(proposed_evaluation, proposed_file);

  std::string proposed_noprior_file = sequence_name + "_ours_noprior.csv";
  LOG(INFO) << "Saving evaluation data into " << proposed_noprior_file;
  save(proposed_noprior_evaluation, proposed_noprior_file);

  std::string iterative_file = sequence_name + "_iterative.csv";
  LOG(INFO) << "Saving evaluation data into " << iterative_file;
  save(iterative_evaluation, iterative_file);

  std::string iterative_noprior_file = sequence_name + "_iterative_noprior.csv";
  LOG(INFO) << "Saving evaluation data into " << iterative_noprior_file;
  save(iterative_noprior_evaluation, iterative_noprior_file);

  std::string mqh_file = sequence_name + "_mqh.csv";
  LOG(INFO) << "Saving evaluation data into " << mqh_file;
  save(mqh_evaluation, mqh_file);

  // write results
  LOG(INFO) << "Saving results data";
  write_results_to_csv(methods, true_scale);

  LOG(INFO) << "done." << std::endl;

  // LOG(INFO) << StringPrintf("Average preintegration time: %.3f",
  //                           1e-3 * static_cast<double>(imu_integration) /
  //                           static_cast<double>(count));
}

int main(int argc, char *argv[])
{

  Eigen::Matrix3d Rcb;
  Rcb << 0.0148655429818000, 0.999557249008000, -0.0257744366974000,
      -0.999880929698000, 0.0149672133247000, 0.00375618835797000,
      0.00414029679422000, 0.0257155299480000, 0.999660727178000;
  Eigen::Vector3d tcb;
  tcb << 0.0652229095355085, -0.0207063854927083, -0.00805460246002837;

  Tcb.linear() = Rcb;
  Tcb.translation() = tcb;

  // Handle help flag
  if (args::HelpRequired(argc, argv))
  {
    args::ShowHelp();
    return 0;
  }

  // Parse input flags
  args::ParseCommandLineNonHelpFlags(&argc, &argv, true);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_log_dir = FLAGS_logs_dir;
  FLAGS_stderrthreshold = 0;
  


  google::InitGoogleLogging(argv[0]);

  // Check number of args
  if (argc - 1 != args::NumArgs())
  {
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
  gap_time = FLAGS_gap_time;
  c_step_size = FLAGS_camera_step_size;
  LOG(INFO) << "Gap time : " << gap_time;
  LOG(INFO) << "camera keyframe step size: " << c_step_size;


  run(ARGS_dataset_dir);

  return 0;
}
