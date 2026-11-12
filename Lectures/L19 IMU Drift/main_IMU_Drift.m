clc, clearvars, close all

% Load Data -- The IMU was laying stationary on a table during the complete
% trial.
% Z axis is aligned with gravity vector (positive pointing towards earth)
% X points along the long axis of the tablet
% DataCollectionDrift_arrays.mat is made from the raw MATLAB Mobile file
% (DataCollectionDrift.mat) by convert_drift_data.m
load("DataCollectionDrift_arrays.mat")  % time, X, Y, Z, start_time
Azimuth = X;
Pitch   = Y;
Roll    = Z;

% Plot Data
figure, sgtitle("Tablet Orientation")
subplot(3,1,1), grid on, hold on
plot(time, Azimuth, '--o'), ylabel('Azimuth [deg]')
subplot(3,1,2), grid on, hold on
plot(time, Pitch, '--o'), ylabel('Pitch [deg]')
subplot(3,1,3), grid on, hold on
plot(time, Roll, '--o'), ylabel('Roll [deg]')
xlabel(['Time [s] (since ' start_time ')'])
