#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd::Identity(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 30;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
  // State dimension
  n_x_ = x_.size();

  // Augumented state dimension
  n_aug_ = n_x_ + 2;

  // lambda
  lambda_ = 3 - n_aug_;

  // number of sigma points
  n_sig_ = 2 * n_aug_ + 1;

  // vector to hold weights
  weights_ = Eigen::VectorXd::Zero(n_sig_);

  // mean of predicted sigma points
  x_pred_ = VectorXd(n_aug_);
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
  if (meas_package.sensor_type_ == MeasurementPackage::LASER){
    UKF::UpdateLidar(meas_package);
    double delta = fabs(meas_package.timestamp_ - time_us_);
    UKF::Prediction(delta);
  }
  else{
    UKF::UpdateRadar(meas_package);
    double delta = fabs(meas_package.timestamp_ - time_us_);
    UKF::Prediction(delta);
  }
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */
  // generate sigma points
  // create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);

  // create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  // create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sig_);
 
  // create augmented mean state
  x_aug.block<5,1>(0,0) = x_;
  x_aug(5,0) = 0;
  x_aug(6,0) = 0;
  // create augmented covariance matrix
  P_aug.block<5,5>(0,0) = P_;
  P_aug(5,5) = pow(std_a_, 2);
  P_aug(6,6) = pow(std_yawdd_, 2);
  // create square root matrix
  MatrixXd A = P_aug.llt().matrixL();
  
  // create augmented sigma points
  Xsig_aug.block<7,1>(0,0) = x_aug;
  Xsig_aug.block<7,7>(0,1) = (sqrt(lambda_ + n_aug_)*A).colwise() + x_aug;
  Xsig_aug.block<7,7>(0,8) = (-sqrt(lambda_ + n_aug_)*A).colwise() + x_aug;

  // predict sigma points
  for (int i = 0; i < n_sig_; i++){
      // predict sigma points
      float v = Xsig_aug.col(i)(2), 
            yaw = Xsig_aug.col(i)(3),
            yaw_rate = Xsig_aug.col(i)(4),
            nu_a = Xsig_aug.col(i)(5),
            nu_psi = Xsig_aug.col(i)(6);
            
      VectorXd noise_vec(5, 1), increment(5, 1);
      
      noise_vec << 0.5 * pow(delta_t, 2) * cos(yaw) * nu_a,
                   0.5 * pow(delta_t, 2) * sin(yaw) * nu_a,
                   delta_t * nu_a,
                   0.5 * pow(delta_t, 2) * nu_psi,
                   delta_t * nu_psi;
                   
      // avoid division by zero
      if (Xsig_aug.col(i)(4) != 0){
        increment <<  (v/yaw_rate) * (sin(yaw + yaw_rate * delta_t) - sin(yaw)),
                      (v/yaw_rate) * (cos(yaw) - cos(yaw + yaw_rate * delta_t)),
                      0,
                      yaw_rate * delta_t,
                      0;
      }
      else{
          increment << v * cos(yaw) * delta_t,
                       v * sin(yaw) * delta_t,
                       0,
                       0,
                       0;
      }
      // write predicted sigma points into the right column
      Xsig_pred_.col(i) = Xsig_aug.col(i).block<5,1>(0,0) + increment + noise_vec;
  }

  // predict mean and covariance
  for (int i = 0; i < n_aug_ ; i++){
    if (i == 0)
      weights_(i) = lambda_ / (lambda_ + n_aug_);
    else
      weights_(i) = 0.5/(lambda_ + n_aug_);
  }
  // predict state mean
  for (int i = 0; i < n_aug_; i++){
      x_pred_ += weights_(i) * Xsig_pred_.col(i);
  }
  // predict state covariance matrix
  for (int i = 0; i < n_aug_; i++){
      VectorXd tmp = Xsig_pred_.col(i) - x_pred_;
      P_ += weights_(i) * tmp * tmp.transpose();
  }
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
  if (is_initialized_ == false){
    is_initialized_ = true;

    x_ << meas_package.raw_measurements_(0), //px
          meas_package.raw_measurements_(1), //py
          0,                                 //vel_abs
          0,                                 //yaw_angle
          0;                                 //yaw_rate
    
    time_us_ = meas_package.timestamp_;
  }
  else{
    // create matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd(2, n_aug_);

    // mean predicted measurement
    VectorXd z_pred = VectorXd(2);
    
    // measurement covariance matrix S
    MatrixXd S = MatrixXd(2, 2);

    Zsig.setZero();
    Zsig.block<1, 15>(0,0) = Xsig_pred_.block<1,15>(0,0);
    Zsig.block<1, 15>(1,0) = Xsig_pred_.block<1,15>(1,0);   
    
    // calculate mean predicted measurement
    for (int i =0; i < Zsig.cols(); i++){
      z_pred += weights_(i) * Zsig.col(i);
    }
      
    // calculate innovation covariance matrix S
    MatrixXd R(2, 2);
    R.diagonal() << pow(std_laspx_,2),
                    pow(std_laspy_,2);
                    
    for (int i =0; i < Zsig.cols(); i++){
      VectorXd tmp = Zsig.col(i) - z_pred;
      S += weights_(i) * tmp * tmp.transpose();
    }
    S += R;

    // create matrix for cross correlation Tc state_dimension x measurement_dimension
    MatrixXd Tc = MatrixXd(n_x_, 2);
    
    for (int i = 0; i < Zsig.cols(); i++){
      Tc += weights_(i) * (Xsig_pred_.col(i) - x_pred_) * (Zsig.col(i) - z_pred).transpose();
    }
    // calculate Kalman gain K;
    MatrixXd K = Tc * S.inverse();
    
    // update state mean and covariance matrix
    x_ += K * (meas_package.raw_measurements_ - z_pred);
    P_ -= K * S * K.transpose();
  }
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
  if (is_initialized_ == false){
    is_initialized_ = true;
    // get values from the raw measurement
    float r = meas_package.raw_measurements_(0);
    float theta = meas_package.raw_measurements_(1);
    float vel = meas_package.raw_measurements_(2);
    // compute the initial state vector
    x_ << r * cos(theta), //px
          r * sin(theta), //py
          vel,            //vel_abs // TODO may have to change the sign 
          0,              //yaw_angle
          0;              //yaw_rate
    
    time_us_ = meas_package.timestamp_;
  }
  else{
    // create matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd(3, n_aug_);

    // mean predicted measurement
    VectorXd z_pred = VectorXd(3);
    
    // measurement covariance matrix S
    MatrixXd S = MatrixXd(3, 3);
    
    for(int i = 0; i < n_aug_; i++){
      float px       = Xsig_pred_(0, i);
      float py       = Xsig_pred_(1, i);
      float v        = Xsig_pred_(2, i);
      float yaw      = Xsig_pred_(3, i);
      float yaw_rate = Xsig_pred_(4, i);
      
      float radial_distance = sqrt(pow(px, 2) + pow(py, 2));
      Zsig.col(i) << radial_distance,
                     atan(py/px),
                     (px * cos(yaw) * v + py * sin(yaw) * v)/radial_distance;
    }
    
    // calculate mean predicted measurement
    for (int i =0; i < Zsig.cols(); i++){
      z_pred += weights_(i) * Zsig.col(i);
    }
      
    // calculate innovation covariance matrix S
    MatrixXd R(3, 3);
    R.diagonal() << pow(std_radr_,2),
                    pow(std_radphi_,2),
                    pow(std_radrd_,2);
                    
    for (int i =0; i < Zsig.cols(); i++){
      VectorXd tmp = Zsig.col(i) - z_pred;
      S += weights_(i) * tmp * tmp.transpose();
    }
    S += R;

    // create matrix for cross correlation Tc state_dimension x measurement_dimension
    MatrixXd Tc = MatrixXd(n_x_, 3);
    
    for (int i = 0; i < Zsig.cols(); i++){
      Tc += weights_(i) * (Xsig_pred_.col(i) - x_pred_) * (Zsig.col(i) - z_pred).transpose();
    }
    // calculate Kalman gain K;
    MatrixXd K = Tc * S.inverse();
    
    // update state mean and covariance matrix
    x_ += K * (meas_package.raw_measurements_ - z_pred);
    P_ -= K * S * K.transpose();
  }
}