%% Truncated Fourier series fit of the ankle angle (least squares)
% Edgar Bolivar
%
% Fits one gait cycle of the ankle angle (Winter, 1983) with a truncated
% Fourier series of order j:
%
%   ankle(t) ~ sum_{k=1..j} a_k*sin(k*w*t) + b_k*cos(k*w*t),  w = 2*pi/T
%
% The model is linear in the coefficients p = [a_1 b_1 ... a_j b_j]', so
% stacking one row per sample gives ankle ~ R*p, and the best fit is the
% least-squares problem  min_p ||ankle - R*p||_2, solved here with "\".
%
% Requires: base MATLAB only.
% Output:   a figure overlaying the data and the Fourier fit.
% Try:      change j. Low orders smooth the curve; high orders chase every
%           sample (overfitting). Keep 2*j <= n so the problem stays
%           well posed (more samples than unknowns).
% Note:     there is no constant (cos(0*w*t)) column, so the fit cannot
%           represent the mean ankle angle. Adding a column of ones is a
%           good exercise.

clearvars, close all
load('Winter1983_Ankle.mat', "ankle", "time")   % column vectors, one gait cycle

% Parameters of the fit
w = 2*pi/time(end);             %Fundamental frequency (one cycle = one period)
j = 8;                         %Order of the fit - Change it!
n = length(ankle);              %Number of points
R = zeros(length(ankle), 2*j);  %Regressor matrix (R) -- ankle ~ R*p

% Create regressor matrix: row i = [sin(w t_i) cos(w t_i) ... sin(j w t_i) cos(j w t_i)]
for i = 1:n
    for ij = 1:j
        R(i, ij*2-1) = sin(w*ij*time(i));
        R(i, ij*2)   = cos(w*ij*time(i));
    end
end

% Get the coefficients that multiply the sinusoids to create the fit
g = R\ankle;        %Least squares power!
ankleFour = R*g;

% Compare results with a plot
figure, hold on, grid on, title("Ankle Angle")
xlabel("Time [seconds]")
ylabel("Ankle Angle [deg] -- Positive Dorsiflexion")
plot(time, ankle, "--o")
plot(time, ankleFour)
legend("Dataset", "Fourier")
