% Convert the raw MATLAB Mobile recording into plain arrays
%
% MATLAB Mobile saves sensor logs as a *timetable* (here: Orientation), whose
% time column is a MATLAB *datetime*. Both are MATLAB objects, and Python
% (scipy.io.loadmat) cannot read them. This script re-saves the same data as
% plain numeric arrays, which both MATLAB and Python can read:
%
%   DataCollectionDrift.mat         (raw, timetable)  -->  this script  -->
%   DataCollectionDrift_arrays.mat  (time, X, Y, Z, start_time)
%
% main_IMU_Drift.m and main_IMU_Drift.py both read the *_arrays.mat file.
% Run this script again only if you record new data with MATLAB Mobile.
clc, clearvars

% The arrays file is already in the repo. MATLAB writes the current date into
% every MAT-file it saves, so regenerating it would change the file even if
% the numbers are the same. Delete it first if you really want to regenerate.
if isfile("DataCollectionDrift_arrays.mat")
    disp("DataCollectionDrift_arrays.mat already exists; delete it first to regenerate it.")
    return
end

% Load the raw recording -- creates the timetable "Orientation" with columns
% Timestamp (datetime), X, Y, Z (orientation angles in degrees)
load("DataCollectionDrift.mat")

% Time: convert absolute clock times into seconds elapsed since the first
% sample (a plain double array)
time = seconds(Orientation.Timestamp - Orientation.Timestamp(1));

% Orientation angles [deg]: MATLAB Mobile reports X = azimuth, Y = pitch,
% Z = roll
X = Orientation.X;
Y = Orientation.Y;
Z = Orientation.Z;

% Keep the clock time of the first sample as text, so it is not lost
start_time = char(Orientation.Timestamp(1), 'yyyy-MM-dd HH:mm:ss.SSS');

% Save as a version-7 MAT-file (the newer -v7.3 is HDF5, which loadmat can't read)
save("DataCollectionDrift_arrays.mat", "time", "X", "Y", "Z", "start_time", "-v7")
fprintf("Saved %d samples (%.1f s) starting at %s\n", numel(time), time(end), start_time)
