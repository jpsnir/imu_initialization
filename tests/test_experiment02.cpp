#include <iostream>
#include <gtest/gtest.h>
#include <experiment02.h>

using namespace std;

TEST(find_closest, upper_bound_test){

};
TEST(read_file, cam_vs_droid_test)
{
    fs::path data_path = "./../../data/MH_01_easy/";
    string path1 = (data_path / "droid_slam" / "KeyFrameTrajectory.txt").string();
    string path2 = (data_path / "KeyFrameTrajectory.txt").string();

    Trajectory traj1 = read_file_TUM(path1);
    Trajectory traj2 = read_file_TUM(path2);

    vector<int> indices = {10, 50, 100, 150, 200};
    int count = 0;
    uint64_t first_timestamp = traj2[0].timestamp;
    std::cout << std::setprecision(10);
    int sum = 0;
    for (unsigned j = 0; j < indices.size(); j++)
    {
        int id_2 = indices[j];
        for (unsigned i = 0; i < traj1.size(); i++)
        {
            // check if we can find timestamps close to the other one.
            if (abs(traj2[id_2].timestamp < traj1[i].timestamp) < 0.1)
            {

                // cout << "Time stamp found : T2_t" << j << "-> " << traj2[indices[j]].timestamp
                //           << "; T1_t" << i << "-> " << traj1[i].timestamp << endl;
            }

            if (traj1[i].timestamp > first_timestamp && traj1[i].timestamp < traj2[id_2].timestamp)
            {
                count++;
            }
        }
        std::cout << "Number of timestamps of traj 1 between timestamps: T2_t: "
                  << first_timestamp << " and " << traj2[id_2].timestamp << "-> " << count
                  << endl;

        first_timestamp = traj2[id_2].timestamp;
        sum += count;
        count = 0;
    }
    std::cout << "Number of camera poses within timetamps - cam - droid in cam - orb: "
              << sum << endl;
}

TEST(read_file, gt_vs_cam_traj_test)
{
    fs::path data_path = "./../../data/MH_01_easy/";
    string path1 = (data_path / "droid_slam" / "KeyFrameTrajectory.txt").string();
    string path2 = (data_path / "state_groundtruth_estimate0" / "data.csv").string();
    string path3 = (data_path / "KeyFrameTrajectory.txt").string();

    Trajectory traj1 = read_file_TUM(path1);
    Trajectory traj2 = read_file_TUM(path3);

    std::vector<io::state_t> gt = io::read_file<io::state_t>(path2);

    unsigned max_id = 36000;
    std::cout << " size of gt trajectory :" << gt.size() << endl;
    int count = 0;
    uint64_t first_timestamp = gt[0].timestamp;
    std::cout << std::setprecision(10);

    list<Trajectory> cam_trajectories = {traj1, traj2};
    int t_id = 0;
    for (auto traj : cam_trajectories)
    {
        t_id++;
        int sum = 0;
        for (unsigned id_2 = 2000; id_2 < max_id; id_2 = id_2 + 2000)
        {
            for (unsigned i = 0; i < traj.size(); i++)
            {
                // check if the trajectory timetamp are within ground truth timestamps.
                if (traj[i].timestamp * 1e9 > first_timestamp && traj[i].timestamp * 1e9 < gt[id_2].timestamp)
                {
                    count++;
                }
            }
            std::cout << "Number of timestamps of traj 1 between timestamps: T2_t: "
                      << first_timestamp << " and " << gt[id_2].timestamp << "-> " << count
                      << endl;

            first_timestamp = gt[id_2].timestamp;
            sum += count;
            count = 0;
        }
        std::cout << "Trj id : " << t_id << ":"
                  << "Number of camera poses within timetamps: "
                  << gt[0].timestamp << " and "
                  << gt[max_id].timestamp << " : "
                  << sum << endl;
    }
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
