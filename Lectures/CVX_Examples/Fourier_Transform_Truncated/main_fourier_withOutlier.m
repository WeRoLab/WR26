%% Truncated Fourier series fit with an outlier (CVX)
% Edgar Bolivar
%
% Same fit as main_fourier.m, but one sample is corrupted by +90 deg (a
% marker swap, say) and the problem is written in CVX instead of "\":
%
%   minimize ||ankle - M*x||_p
%
% One bad sample among 1001 moves the L2 (least-squares) fit only ~1.4 deg.
% A burst of bad samples is different: squared residuals punish large
% errors heavily, so the L2 fit bends toward them.
%
% Requires: CVX (https://cvxr.com/cvx/download/) on the MATLAB path.
% Output:   a figure overlaying the corrupted data and the fit.
% Try:      corrupt a burst, ankle(100:150) = ankle(100:150)+90, and compare
%           norm(..., 2) with norm(..., 1). The L2 fit shifts by up to ~66
%           deg; the L1 fit lets those samples keep large residuals and stays
%           within ~5 deg of its clean-data fit. (norm(..., Inf) is the
%           opposite extreme.)

clc, clearvars, close all
load('Winter1983_Ankle.mat')    % loads "ankle" and "time"

% Introduce outlier
ankle(100) = ankle(100)+90;

% Create regressor matrix (same construction as R in main_fourier.m)
w = 2*pi/time(end);
j = 8;              %Order of the fit
n = length(ankle);  %Number of points
M = zeros(length(ankle), 2*j);

for i = 1:n
    for ij = 1:j
        M(i, ij*2-1) = sin(w*ij*time(i));
        M(i, ij*2)   = cos(w*ij*time(i));
    end
end

% Fit the coefficients x; change the 2 to 1 for a robust (L1) fit
cvx_begin quiet
    variables x(2*j)
    minimize norm(ankle - M*x, 2)
cvx_end

figure, hold on, grid on, title("Ankle Angle")
xlabel("Time [seconds]"), ylabel("Ankle Angle [deg] - Pos. Dorsiflexion")
plot(time, ankle, "--o")
plot(time, M*x)
legend("Dataset", "Fourier")
