#ifndef UKF_H
#define UKF_H

#include "Eigen/Dense"
#include "measurement_package.h"

class UKF {
 public:
  /**
   * Constructor
   */
  UKF();

  /**
   * Destructor
   */
  virtual ~UKF();
  void InitializeState(const MeasurementPackage& meas_package);
  void InitializeFromLaser(const MeasurementPackage& meas_package); 
  void InitializeFromRadar(const MeasurementPackage& meas_package);
  /**
   * ProcessMeasurement
   * @param meas_package The latest measurement data of either radar or laser
   */
  void ProcessMeasurement(const MeasurementPackage& meas_package);

   // Generates augmented sigma points for the prediction step.
  void GenerateAugmentedSigmaPoints(Eigen::MatrixXd& Xsig_aug);
  // Predicts sigma points for the next state.
  void PredictSigmaPoints(const Eigen::MatrixXd& Xsig_aug, double delta_t);
  // Predicts the mean and covariance of the sigma points to update the state.
  void PredictMeanAndCovariance();
  // Normalizes an angle to be within -PI and PI.
  double NormalizeAngle(double angle);
  /**
   * Prediction Predicts sigma points, the state, and the state covariance
   * matrix
   * @param delta_t Time between k and k+1 in s
   */
  void Prediction(double delta_t);
  
/**
 * Calculate the measurement covariance matrix S for the update step.
 * @return MatrixXd The calculated measurement covariance matrix S.
 */
Eigen::MatrixXd CalculateMeasurementCovarianceMatrixS(const Eigen::MatrixXd& Zsig, const Eigen::VectorXd& z_pred, int n_z);
/**
 * Updates the state mean and covariance matrix using the incoming measurement.
 * This function calculates the cross-correlation matrix, the Kalman gain, and then updates
 * the state mean and covariance matrix accordingly.
 * @param z The incoming measurement vector.
 * @param Zsig The matrix of sigma points in measurement space.
 * @param z_pred The mean predicted measurement vector.
 * @param S The measurement covariance matrix.
 */
void UpdateState(const Eigen::VectorXd& z, const Eigen::MatrixXd& Zsig, const Eigen::VectorXd& z_pred, const Eigen::MatrixXd& S);

  /**
   * Updates the state and the state covariance matrix using a laser measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateLidar(MeasurementPackage meas_package);

  /**
   * Updates the state and the state covariance matrix using a radar measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateRadar(MeasurementPackage meas_package);
/**
 * Transforms sigma points into measurement space.
 * @param Xsig_pred The sigma points in state space
 * @return Matrix of sigma points in measurement space, dimensions depend on the measurement model.
 */
  Eigen::MatrixXd TransformSigmaPointsIntoMeasurementSpace(const Eigen::MatrixXd &Xsig_pred);

/**
 * Calculates the mean predicted measurement from sigma points in measurement space.
 * @param Zsig The sigma points in measurement space, dimensions vary according to measurement model 
 * @return Vector of predicted measurements, dimension depends on the measurement model.
 */
  Eigen::VectorXd CalculatePredictedMeasurement(const Eigen::MatrixXd &Zsig);

/**
 * Calculates the measurement covariance matrix S using sigma points in measurement space.
 * @param Zsig The sigma points in measurement space.
 * @param z_pred The predicted measurement vector.
 * @return The measurement covariance matrix S.
 */
  Eigen::MatrixXd CalculateMeasurementCovarianceMatrix(const Eigen::MatrixXd &Zsig, const Eigen::VectorXd &z_pred);

/**
 * Calculates the cross-correlation matrix between the sigma points in state space and the sigma points in measurement space.
 * @param Zsig The sigma points in measurement space.
 * @param z_pred The mean predicted measurement.
 * @param x_pred The mean predicted state.
 * @return The cross-correlation matrix Tc.
 */
  Eigen::MatrixXd CalculateCrossCorrelationMatrix(const Eigen::MatrixXd &Zsig, const Eigen::VectorXd &z_pred, const Eigen::VectorXd &x_pred);

  // initially set to false, set to true in first call of ProcessMeasurement
  bool is_initialized_;

  // if this is false, laser measurements will be ignored (except for init)
  bool use_laser_;

  // if this is false, radar measurements will be ignored (except for init)
  bool use_radar_;

  // state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  Eigen::VectorXd x_;

  // state covariance matrix
  Eigen::MatrixXd P_;

  // predicted sigma points matrix
  Eigen::MatrixXd Xsig_pred_;

  // time when the state is true, in us
  long long time_us_;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a_;

  // Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd_;

  // Laser measurement noise standard deviation position1 in m
  double std_laspx_;

  // Laser measurement noise standard deviation position2 in m
  double std_laspy_;

  // Radar measurement noise standard deviation radius in m
  double std_radr_;

  // Radar measurement noise standard deviation angle in rad
  double std_radphi_;

  // Radar measurement noise standard deviation radius change in m/s
  double std_radrd_ ;

  // Weights of sigma points
  Eigen::VectorXd weights_;

  // State dimension
  int n_x_;

  // Augmented state dimension
  int n_aug_;

  // Sigma point spreading parameter
  double lambda_;
};

#endif  // UKF_H