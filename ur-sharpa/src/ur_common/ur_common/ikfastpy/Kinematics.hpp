namespace robots {
    class Kinematics {
        public: int num_of_joints, num_free_parameters;
        Kinematics();
        ~Kinematics();
        std::vector<double> forward(std::vector<double> joint_config);
        std::vector<double> inverse(std::vector<double> ee_pose);
    };
}
