#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() 
    : is_initialized_(false), // Initially set to false, updated in first call of ProcessMeasurement
      use_laser_(true),       // Enable laser measurements by default
      use_radar_(true),       // Enable radar measurements by default
      n_x_(5),                // State dimension
      n_aug_(7),              // Augmented state dimension
      lambda_(3 - n_aug_),    // Sigma point spreading parameter
      time_us_(0.0),          // Initial time in microseconds
      std_a_(2.0),            // Process noise std dev for longitudinal acceleration
      std_yawdd_(2.0),        // Process noise std dev for yaw acceleration
      std_laspx_(0.15),       // Laser measurement noise std dev for position in x
      std_laspy_(0.15),       // Laser measurement noise std dev for position in y
      std_radr_(0.3),         // Radar measurement noise std dev for radius
      std_radphi_(0.03),      // Radar measurement noise std dev for angle
      std_radrd_(0.3)         // Radar measurement noise std dev for radius change
{
    // Initialize state vector and covariance matrix with zeros
    x_ = VectorXd::Zero(n_x_);
    P_ = MatrixXd::Zero(n_x_, n_x_);

    // Initialize predicted sigma points matrix with zeros
    Xsig_pred_ = MatrixXd::Zero(n_x_, 2 * n_aug_ + 1);

    // Initialize weights of sigma points
    weights_ = VectorXd(2 * n_aug_ + 1);
    weights_(0) = lambda_ / (lambda_ + n_aug_);
    double weight = 0.5 / (lambda_ + n_aug_);
    for (int i = 1; i < 2 * n_aug_ + 1; ++i) {
        weights_(i) = weight;
    }
}

UKF::~UKF() {}
/*
* The ProcessMeasurement function has 3 helpers
* 1.)InitializeState
* 2.)InitializeFromLaser
* 3.) InitializeFromRadar
*/
void UKF::InitializeState(const MeasurementPackage& meas_package) {
    // Check the type of sensor and initialize the state accordingly
    if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
        InitializeFromLaser(meas_package);
    } else if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
        InitializeFromRadar(meas_package);
    } else {
        std::cerr << "UKF::ProcessMeasurement() error: Invalid or disabled sensor type " 
                  << meas_package.sensor_type_ << std::endl;
        return;
    }
}

void UKF::InitializeFromLaser(const MeasurementPackage& meas_package) {
  	//std::cout << "Initialized from laser\n" << std::endl; // For Debug
    // Set the state with the initial location and zero velocity
    x_ << meas_package.raw_measurements_[0], 
          meas_package.raw_measurements_[1], 
          0, 0, 0;

    P_ << std_laspx_ * std_laspx_, 0, 0, 0, 0,
          0, std_laspy_ * std_laspy_, 0, 0, 0,
          0, 0, 1, 0, 0,
          0, 0, 0, 1, 0,
          0, 0, 0, 0, 1;
    is_initialized_ = true; // Set initialization flag
    time_us_ = meas_package.timestamp_; // set time stamp immediately after init
}

void UKF::InitializeFromRadar(const MeasurementPackage& meas_package) {
    //std::cout << "Initialized from radar\n" << std::endl;  // For Debug

    // Convert radar from polar to cartesian coordinates and initialize state
    double rho = meas_package.raw_measurements_[0];
    double phi = meas_package.raw_measurements_[1];
    double rho_dot = meas_package.raw_measurements_[2]; // Not used for initial state

    x_ << rho * cos(phi), 
          rho * sin(phi), 
          0, 0, 0;

    P_ << std_radr_ * std_radr_, 0, 0, 0, 0,
          0, std_radr_ * std_radr_, 0, 0, 0,
          0, 0, std_radrd_ * std_radrd_, 0, 0,
          0, 0, 0, std_radphi_ * std_radphi_, 0,
          0, 0, 0, 0, 1;
    is_initialized_ = true; // Set initialization flag
    time_us_ = meas_package.timestamp_; // set time stamp immediately after init
}

void UKF::ProcessMeasurement(const MeasurementPackage& meas_package) {
    // Initialization check
    if (!is_initialized_) {
        InitializeState(meas_package);
        return; // Early return since no prediction or update is needed on the first measurement
    }

    // Time elapsed between the current and previous measurements (in seconds)
    double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
    time_us_ = meas_package.timestamp_; // Update last timestamp

    // Prediction step
    Prediction(dt);

    // Update step based on the type of measurement
    switch (meas_package.sensor_type_) {
        case MeasurementPackage::LASER:
            if (use_laser_) UpdateLidar(meas_package);
            break;
        case MeasurementPackage::RADAR:
            if (use_radar_) UpdateRadar(meas_package);
            break;
        default:
            // Log error for unsupported sensor type
            std::cerr << "UKF::ProcessMeasurement() error: Unsupported sensor type " 
                      << meas_package.sensor_type_ << std::endl;
            break;
    }
}

void UKF::Prediction(double delta_t) {
	MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
	GenerateAugmentedSigmaPoints(Xsig_aug); // Step 1: Generate augmented sigma points
	PredictSigmaPoints(Xsig_aug, delta_t);   // Step 2: Predict sigma points for the next state
  	PredictMeanAndCovariance();              // Step 3: Predict the mean and covariance
}

void UKF::GenerateAugmentedSigmaPoints(Eigen::MatrixXd& Xsig_aug) {
	VectorXd x_aug = VectorXd::Zero(n_aug_); // Create augmented mean state
  	x_aug.head(n_x_) = x_;                   // Set first part of x_aug to the current state

  	// Create augmented covariance matrix
  	MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);
  	P_aug.topLeftCorner(n_x_, n_x_) = P_;    // Current state covariance
  	P_aug(n_x_, n_x_) = std_a_ * std_a_;     // Process noise standard deviation for acceleration
  	P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_ * std_yawdd_; // Process noise std dev for yaw acceleration

  	// Calculate square root of P_aug
 	 MatrixXd L = P_aug.llt().matrixL();

  	// Set first column of sigma point matrix
  	Xsig_aug.col(0) = x_aug;
  	// Set remaining sigma points
  	for (int i = 0; i < n_aug_; i++)
    {
    	Xsig_aug.col(i + 1)         = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
    	Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
  	}
}

void UKF::PredictSigmaPoints(const Eigen::MatrixXd& Xsig_aug, double delta_t) {
    // Loop through all sigma points
    for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
        // Extract values for readability
        double position_x = Xsig_aug(0, i);
        double position_y = Xsig_aug(1, i);
        double velocity = Xsig_aug(2, i);
        double yaw_angle = Xsig_aug(3, i);
        double yaw_rate = Xsig_aug(4, i);
        double linear_acc_noise = Xsig_aug(5, i);
        double angular_acc_noise = Xsig_aug(6, i);

        // Predicted state values
        double predicted_x, predicted_y;

        // Avoid division by zero for yaw rate
        if (fabs(yaw_rate) > 0.001) {
            predicted_x = position_x + velocity / yaw_rate * (sin(yaw_angle + yaw_rate * delta_t) - sin(yaw_angle));
            predicted_y = position_y + velocity / yaw_rate * (cos(yaw_angle) - cos(yaw_angle + yaw_rate * delta_t));
        } else {
            predicted_x = position_x + velocity * delta_t * cos(yaw_angle);
            predicted_y = position_y + velocity * delta_t * sin(yaw_angle);
        }

        // Incorporate noise
        double predicted_velocity = velocity + linear_acc_noise * delta_t;

        double predicted_yaw = yaw_angle + yaw_rate * delta_t + 0.5 * delta_t * delta_t * angular_acc_noise;
        double predicted_yaw_rate = yaw_rate + angular_acc_noise * delta_t;

        // Normalize the yaw angle
        predicted_yaw = NormalizeAngle(predicted_yaw);

        // Set the predicted sigma point
        Xsig_pred_(0, i) = predicted_x;
        Xsig_pred_(1, i) = predicted_y;
        Xsig_pred_(2, i) = predicted_velocity;
        Xsig_pred_(3, i) = predicted_yaw;
        Xsig_pred_(4, i) = predicted_yaw_rate;
    }
}


void UKF::PredictMeanAndCovariance() {
    // Initialize state mean vector
    x_.setZero();

    // Calculate predicted state mean
    for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
        x_ += weights_(i) * Xsig_pred_.col(i);
    }

    // Initialize state covariance matrix
    P_.setZero();

    // Calculate predicted state covariance matrix
    for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
        // Calculate state difference and normalize yaw angle
        Eigen::VectorXd state_diff = Xsig_pred_.col(i) - x_;
        state_diff(3) = NormalizeAngle(state_diff(3));

        // Update state covariance matrix
        P_ += weights_(i) * state_diff * state_diff.transpose();
    }
}

double UKF::NormalizeAngle(double angle) {
	// Ensure the angle is within -PI to PI
	while (angle > M_PI) angle -= 2. * M_PI;
	while (angle < -M_PI) angle += 2. * M_PI;
	return angle;
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
    int n_z = 2; // Measurement dimension: lidar measures x, y

    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
    // Transform sigma points into measurement space
    for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
        double p_x = Xsig_pred_(0, i);
        double p_y = Xsig_pred_(1, i);
        Zsig(0, i) = p_x;
        Zsig(1, i) = p_y;
    }

    // Calculate mean predicted measurement
    VectorXd z_pred = CalculatePredictedMeasurement(Zsig);

    // Calculate innovation covariance matrix S
    MatrixXd S = CalculateMeasurementCovarianceMatrixS(Zsig, z_pred, n_z);

    // Update state mean and covariance matrix
    UpdateState(meas_package.raw_measurements_, Zsig, z_pred, S);
}

// Updates the state with radar measurements.
void UKF::UpdateRadar(MeasurementPackage meas_package) {
    // Transform sigma points into measurement space for radar.
    MatrixXd Zsig = TransformSigmaPointsIntoMeasurementSpace(Xsig_pred_);

    // Calculate mean predicted measurement for radar.
    VectorXd z_pred = CalculatePredictedMeasurement(Zsig);

    // Calculate innovation covariance matrix S for radar.
    MatrixXd S = CalculateMeasurementCovarianceMatrix(Zsig, z_pred);

    // Update the state based on radar measurement.
    UpdateState(meas_package.raw_measurements_, Zsig, z_pred, S);  
}

// Calculates the mean predicted measurement from sigma points in measurement space.
Eigen::VectorXd UKF::CalculatePredictedMeasurement(const Eigen::MatrixXd& Zsig) {
    int n_z = Zsig.rows(); // Measurement space dimension
    VectorXd z_pred = VectorXd::Zero(n_z);
    // Iterate over sigma points to calculate weighted mean.
    for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
        z_pred += weights_(i) * Zsig.col(i);
    }
    return z_pred;
}

// Calculates the measurement covariance matrix S for a given set of sigma points in measurement space.
Eigen::MatrixXd UKF::CalculateMeasurementCovarianceMatrixS(const Eigen::MatrixXd& Zsig, const Eigen::VectorXd& z_pred, int n_z) {
    MatrixXd S = MatrixXd::Zero(n_z, n_z);
    // Add measurement noise covariance matrix
    MatrixXd R = MatrixXd(n_z, n_z);
    R << std_laspx_ * std_laspx_, 0,
         0, std_laspy_ * std_laspy_;
    // Iterate over each sigma point to accumulate the covariance information.
    for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
        VectorXd z_diff = Zsig.col(i) - z_pred;
         z_diff(1) = NormalizeAngle(z_diff(1)); 
        S += weights_(i) * z_diff * z_diff.transpose();
    }
    S += R;
    return S;
}

MatrixXd UKF::TransformSigmaPointsIntoMeasurementSpace(const Eigen::MatrixXd &Xsig_pred) {
    int measure_dim = 3; // r, phi, r_dot
    MatrixXd Zsig = MatrixXd(measure_dim, 2 * n_aug_ + 1);

    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        double p_x = Xsig_pred(0, i), p_y = Xsig_pred(1, i);
        double vel = Xsig_pred(2, i), yaw = Xsig_pred(3, i);
        double v_x = cos(yaw) * vel, v_y = sin(yaw) * vel;

        Zsig(0, i) = sqrt(p_x * p_x + p_y * p_y); // r
        Zsig(1, i) = atan2(p_y, p_x); // phi
        Zsig(2, i) = (p_x * v_x + p_y * v_y) / std::max(sqrt(p_x * p_x + p_y * p_y), 0.0001); // r_dot
    }
    return Zsig;
}

MatrixXd UKF::CalculateMeasurementCovarianceMatrix(const Eigen::MatrixXd &Zsig, const Eigen::VectorXd &z_pred) {
    int measure_dim = 3;
    MatrixXd S = MatrixXd::Zero(measure_dim, measure_dim);
    MatrixXd R = MatrixXd(measure_dim, measure_dim);
    R << std_radr_ * std_radr_, 0, 0,
         0, std_radphi_ * std_radphi_, 0,
         0, 0, std_radrd_ * std_radrd_;

    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        VectorXd z_diff = Zsig.col(i) - z_pred;
        z_diff(1) = NormalizeAngle(z_diff(1));
        S += weights_(i) * z_diff * z_diff.transpose();
    }

    S += R;
    return S;
}

MatrixXd UKF::CalculateCrossCorrelationMatrix(const Eigen::MatrixXd &Zsig, const Eigen::VectorXd &z_pred, const Eigen::VectorXd &x_pred) {
    int n_z = z_pred.size();
    MatrixXd Tc = MatrixXd::Zero(n_x_, n_z);

    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        VectorXd z_diff = Zsig.col(i) - z_pred;
        z_diff(1) = NormalizeAngle(z_diff(1));

        VectorXd x_diff = Xsig_pred_.col(i) - x_pred;
        x_diff(3) = NormalizeAngle(x_diff(3));

        Tc += weights_(i) * x_diff * z_diff.transpose();
    }

    return Tc;
}

void UKF::UpdateState(const Eigen::VectorXd& z, const Eigen::MatrixXd& Zsig, const Eigen::VectorXd& z_pred, const Eigen::MatrixXd& S) {
    MatrixXd Tc = CalculateCrossCorrelationMatrix(Zsig, z_pred, x_);

    // Calculate Kalman gain K
    MatrixXd K = Tc * S.inverse();

    // Update state mean and covariance matrix
    VectorXd z_diff = z - z_pred;
    z_diff(1) = NormalizeAngle(z_diff(1)); // Assuming angle normalization is necessary for the measurement residual

    x_ += K * z_diff;
    P_ -= K * S * K.transpose();
}
