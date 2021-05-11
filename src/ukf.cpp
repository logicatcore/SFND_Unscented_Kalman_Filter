#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

double normalizeAngle(double angle) {
    return atan2(sin(angle), cos(angle));
}   

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
  
  P_.diagonal() << 0.1,
                   0.1,
                   3.5,
                   0.25,
                   0.25;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.6;
  
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
  n_x_ = 5;

  // Augumented state dimension
  n_aug_ = n_x_ + 2;

  // lambda
  lambda_ = 3 - n_aug_;

  // number of sigma points
  n_sig_ = 2 * n_aug_ + 1;

  // vector to hold weights
  weights_ = Eigen::VectorXd(n_sig_);

  // predicted sigma points
  Xsig_pred_ = MatrixXd::Zero(n_x_, n_sig_);

  // time difference
  delta_ = 0;

  // time initialisation
  time_us_ = 0.0;

  // measurement matrix
  H_ = MatrixXd(2, n_x_);
  H_ << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0;

  // measurement covariance matrix
  R_ = MatrixXd(2, 2);
  R_ << pow(std_laspx_, 2), 0,
        0, pow(std_laspy_, 2);

  // weights do not change so calculate once
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  double tmp = 0.5 / (lambda_ + n_aug_);
  for (int i = 1; i < n_sig_ ; i++){
    weights_(i) = tmp;
  }
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
  if (meas_package.sensor_type_ == MeasurementPackage::LASER){
    if (is_initialized_ == false && use_laser_){
      time_us_  = meas_package.timestamp_;
      is_initialized_ = true;

      x_ << meas_package.raw_measurements_(0), //px
            meas_package.raw_measurements_(1), //py
            0,                                 //vel_abs
            0,                                 //yaw_angle
            0;                                 //yaw_rate
    }
    else if (use_laser_) {
      if (time_us_ != meas_package.timestamp_){
        delta_ = (meas_package.timestamp_ - time_us_) * 1e-6;
        time_us_ = meas_package.timestamp_;
        UKF::Prediction(delta_);
      }
      UKF::UpdateLidar(meas_package);
    }
  }
  else{
    if (is_initialized_ == false && use_radar_) {
      time_us_  = meas_package.timestamp_;
      is_initialized_ = true;
      // get values from the raw measurement
      float r = meas_package.raw_measurements_(0);
      float theta = meas_package.raw_measurements_(1);
      float vel = meas_package.raw_measurements_(2);
      // compute the initial state vector
      x_ << r * cos(theta),   //px
            r * sin(theta),   //py
            0,                //vel_abs // TODO may have to change the sign 
            theta,            //yaw_angle
            0;                //yaw_rate
    }
    else if (use_radar_) {
      if (time_us_ != meas_package.timestamp_){
        delta_ = (meas_package.timestamp_ - time_us_) * 1e-6;
        time_us_ = meas_package.timestamp_;
        UKF::Prediction(delta_);
      }
      UKF::UpdateRadar(meas_package);
    }
  }
  // std::cout << "Time is: " << time_us_ << "P_ is: " << P_ << std::endl;
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
  x_aug(5) = 0;
  x_aug(6) = 0;
  // create augmented covariance matrix
  P_aug.fill(0);
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
  // VectorXd mean = Xsig_aug.arr
  for (int i = 0; i < n_sig_; i++){
    // predict sigma points
    double px = Xsig_aug(0, i);
    double py = Xsig_aug(1, i);
    double v = Xsig_aug(2, i);
    double yaw = Xsig_aug(3, i);
    double yaw_rate = Xsig_aug(4, i);
    double nu_a = Xsig_aug(5, i);
    double nu_psi = Xsig_aug(6, i);

    double px_tmp(0), py_tmp(0), v_tmp(0), yaw_tmp(0), yawd_tmp(0);

    // increment
    if (fabs(yaw_rate) > 0.001){
      px_tmp = px + (v/yaw_rate) * (sin(yaw + yaw_rate * delta_t) - sin(yaw));
      py_tmp = py + (v/yaw_rate) * (cos(yaw) - cos(yaw + yaw_rate * delta_t));
    }
    else{
      px_tmp = px + v * cos(yaw) * delta_t;
      py_tmp = py + v * sin(yaw) * delta_t;
    }
    yaw_tmp = yaw + yaw_rate * delta_t;                            

    // noise vector
    px_tmp += 0.5 * pow(delta_t, 2) * cos(yaw) * nu_a;
    py_tmp += 0.5 * pow(delta_t, 2) * sin(yaw) * nu_a;
    v_tmp  = v + delta_t * nu_a;
    yaw_tmp = yaw_tmp + 0.5 * pow(delta_t, 2) * nu_psi;
    yawd_tmp = yaw_rate + delta_t * nu_psi;
                  
    // write predicted sigma points into the right column
    Xsig_pred_(0,i) = px_tmp; 
    Xsig_pred_(1,i) = py_tmp;
    Xsig_pred_(2,i) = v_tmp;
    Xsig_pred_(3,i) = yaw_tmp;
    Xsig_pred_(4,i) = yawd_tmp;
  }

  // predict mean and covariance
  // predict state mean
  x_.fill(0.0);
  for (int i = 0; i < n_sig_; i++){
      x_ += weights_(i) * Xsig_pred_.col(i);
  }
  // std::cout << x_ << std::endl;
  // predict state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < n_sig_; i++){
      VectorXd tmp = Xsig_pred_.col(i) - x_;

      // angle normalization
      if (tmp(3) > M_PI || tmp(3) < -M_PI)
        tmp(3) = normalizeAngle(tmp(3));

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
  // handle when two measurements with same time stamp arrive at the beginning
  // first radar and then lidar, no need to average vel, yaw angle, and yaw rate
  if (delta_ == 0){
    x_ << (x_(0) + meas_package.raw_measurements_(0))/2, //px
          (x_(1) + meas_package.raw_measurements_(1))/2, //py
          x_(2),                                         //vel_abs
          x_(3),                                         //yaw_angle
          x_(4);                                         //yaw_rate
    // std::cout << x_ << std::endl;
  }
  else{  
    VectorXd z = VectorXd(2);
    z << meas_package.raw_measurements_(0), 
         meas_package.raw_measurements_(1);

    VectorXd z_pred = H_ * x_;
    VectorXd y = z - z_pred;
    MatrixXd S = H_ * P_ * H_.transpose() + R_;
    MatrixXd K = P_ * H_.transpose() * S.inverse();

    //new estimate
    x_ = x_ + K * y;
    // std::cout << x_ << std::endl;

    MatrixXd I = MatrixXd::Identity(n_x_, n_x_);
    P_ = (I - K * H_) * P_;
  }
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
  // handle when two measurements with same time stamp arrive at the beginning
  // first lidar and then radar, so no need to average velocity
  if (delta_ == 0){
    // get values from the raw measurement
    float r = meas_package.raw_measurements_(0);
    float theta = meas_package.raw_measurements_(1);
    float vel = meas_package.raw_measurements_(2);

    x_ << (x_(0) + r * cos(theta))/2, //px
          (x_(1) + r * sin(theta))/2, //py
          0,                          //vel_abs
          theta,                          //yaw_angle
          0;                          //yaw_rate
    // std::cout << x_ << std::endl;
  }
  else{
    // create matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd(3, n_sig_);

    // mean predicted measurement
    VectorXd z_pred = VectorXd(3);
    
    // measurement covariance matrix S
    MatrixXd S = MatrixXd(3, 3);
    
    for(int i = 0; i < n_sig_; i++){
      float px       = Xsig_pred_(0, i);
      float py       = Xsig_pred_(1, i);
      float v        = Xsig_pred_(2, i);
      float yaw      = Xsig_pred_(3, i);
      float yaw_rate = Xsig_pred_(4, i);
      
      float radial_distance = sqrt(pow(px, 2) + pow(py, 2));
      radial_distance = 0.0001 > radial_distance ? 0.0001:radial_distance;
      Zsig.col(i) << radial_distance,
                     atan2(py, px),
                     (px * cos(yaw) * v + py * sin(yaw) * v)/ radial_distance;
    }
    
    // calculate mean predicted measurement
    z_pred.fill(0);
    for (int i =0; i < n_sig_; i++){
      z_pred += weights_(i) * Zsig.col(i);
    }
      
    // calculate innovation covariance matrix S
    MatrixXd R(3, 3);
    R.fill(0);
    R.diagonal() << pow(std_radr_,2),
                    pow(std_radphi_,2),
                    pow(std_radrd_,2);
    
    S.fill(0.0);  
    for (int i =0; i < n_sig_; i++){
      VectorXd tmp = Zsig.col(i) - z_pred;

      // angle normalization
      if (tmp(1) > M_PI || tmp(1) < -M_PI)
        tmp(1) = normalizeAngle(tmp(1));

      S += weights_(i) * tmp * tmp.transpose();
    }
    S += R;

    // create matrix for cross correlation Tc state_dimension x measurement_dimension
    MatrixXd Tc = MatrixXd(n_x_, 3);
    Tc.fill(0.0);

    for (int i = 0; i < n_sig_; i++){
      // residual
      VectorXd z_diff = Zsig.col(i) - z_pred;
      
      // angle normalization
      if (z_diff(1) > M_PI || z_diff(1) < -M_PI)
        z_diff(1) = normalizeAngle(z_diff(1));

      // state difference
      VectorXd x_diff = Xsig_pred_.col(i) - x_;
      
      // angle normalization
      if (x_diff(3) > M_PI || x_diff(3) < -M_PI)
        x_diff(3) = normalizeAngle(x_diff(3));
    
      Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }
    // calculate Kalman gain K;
    MatrixXd K = Tc * S.inverse();
    
    // residual
    VectorXd z_diff = meas_package.raw_measurements_ - z_pred;

    // angle normalization
    if (z_diff(1) > M_PI || z_diff(1) < -M_PI)
      z_diff(1) = normalizeAngle(z_diff(1));

    // update state mean and covariance matrix
    x_ += K * z_diff;
    // std::cout << x_ << std::endl;
    P_ -= K * S * K.transpose();
  }
}