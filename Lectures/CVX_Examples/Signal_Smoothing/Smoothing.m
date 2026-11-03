%% Smoothing of signal to minimize jerk
% Edgar Bolivar
%
% Removes noise from a sampled signal by finding the curve x that stays
% close to the measurement y while having small jerk (3rd derivative):
%
%   minimize ||x - y||^2 + (alpha*dt^3)^2 * ||D3*x||^2
%
% D3 is a finite-difference matrix, so this is a linear least-squares
% problem: stack [I; alpha*dt^3*D3]*x ~ [y; 0] and solve with "\". The same
% problem is then solved with CVX to compare run times and check that both
% solvers agree.
%
% Requires: base MATLAB. CVX (https://cvxr.com/cvx/download/) is optional;
%           without it the CVX solve is skipped with a warning.
% Output:   two figures (position, and position/velocity/acceleration of
%           the original, noisy and smoothed signals) and printed timings.
% Try:      change alpha (third argument of smoothCVX). Too small keeps the
%           noise; too large flattens the signal. The default 1e-4 is in the
%           "too small" range (the result is almost the noisy signal);
%           alpha = 1 to 10 visibly smooths it. Note how the noise is
%           amplified in the velocity and acceleration of the noisy signal.

clc, clearvars, close all

time    = linspace(0, 2, 1000).';
dt      = time(2) - time(1);
sig     = sin(2*pi*time);                               %Signal
noi     = sig + (rand(1, length(sig)).'-0.5)*0.1;       %Noisy signal (uniform, +/-0.05)

% Smooth the signal (calculate alpha (third parameter) by trial and error)
[smo, D1, D2] = smoothCVX(noi, dt, 10e-5);   % 10e-5 = 1e-4

% Plot and compare
figure, title('Reference signals')
hold on, grid on, xlabel('Time'), ylabel('Position')
plot(time, sig)
plot(time, noi, '--')
plot(time, smo, '--o')
legend('Original signal', 'Noisy signal', 'Smoothed signal')

% D1/D2 drop the first and last two samples, hence time(3:end-2)
figure, sgtitle('Signal smoothing'), subplot(3,1,1)
hold on, grid on, xlabel('Time'), ylabel('Position')
plot(time, sig)
plot(time, noi, '--')
plot(time, smo, '--o')
legend('Original signal', 'Noisy signal', 'Smoothed signal')
subplot(3,1,2)
hold on, grid on, xlabel('Time'), ylabel('Velocity')
plot(time(3:end-2), D1*sig)
plot(time(3:end-2), D1*noi, '--')
plot(time(3:end-2), D1*smo, '--o')
subplot(3,1,3)
hold on, grid on, xlabel('Time'), ylabel('Acceleration')
plot(time(3:end-2), D2*sig)
plot(time(3:end-2), D2*noi, '--')
plot(time(3:end-2), D2*smo, '--o')

function [fil, D1, D2] = smoothCVX(sig, dt, alp)
% Find a set of points that are close to sig but minimize jerk.
% Returns the smoothed signal and the 1st/2nd derivative matrices (for plotting).

%-Create derivative operator Matrix (Center difference)
% Each row applies a 5-point stencil on samples k-2..k+2. Rows whose
% stencil would run off the ends are removed, so D*x has n-4 entries.
n = length(sig);
e = ones(n,1);

% First derivative - 4th order accuracy:  [1 -8 0 8 -1]/(12 dt)
D1 = spdiags([e/12, -2/3*e, 0*e, 2/3*e, -1/12*e], -2:2, n, n);
D1 = D1(3:end-2, :)/dt;

% Second derivative - 4th order accuracy: [-1 16 -30 16 -1]/(12 dt^2)
D2 = spdiags([-e/12, 4/3*e, -5/2*e, 4/3*e, -1/12*e], -2:2, n, n);
D2 = D2(3:end-2, :)/dt^2;

% Third derivative - 2nd order accuracy:  [-1 2 0 -2 1]/(2 dt^3)
D3 = spdiags([-e/2, 1*e, 0*e, -1*e, 1/2*e], -2:2, n, n);
D3 = D3(3:end-2, :)/dt^3;

% Minimize the norm of signal fit and jerk: || [I; alp*dt^3*D3]*x - [sig; 0] ||
[m,~] = size(D3);
A   = [eye(n); (alp*dt^3)*D3];
y   = [sig; zeros(m,1)];

% Solve optimization using built-in MATLAB least squares
tic
fil = A\y;                  % Least squares power!
tMATLAB = toc;

% Solve optimization using CVX (https://cvxr.com/cvx/download/)
try
    tic
    cvx_begin quiet
    variable x(n)
    minimize norm(A*x - y, 2)
    cvx_end
    fil2 = x;
    tCVX = toc;
catch
    warning("Solution using CVX failed. Check that CVX is installed.")
    tCVX = 0;
    fil2 = [];
end

fprintf("Elapsed time MATLAB:\t%3.4f [s]\n", tMATLAB)
fprintf("Elapsed time CVX:\t\t%3.4f [s]\n", tCVX)
% Example output
% Elapsed time MATLAB:	0.0017 [s]
% Elapsed time CVX:		0.5143 [s]

% Both solvers should find the same minimizer
if ~isempty(fil2)
    fprintf("Max |MATLAB - CVX|:\t%3.2e\n", max(abs(fil - fil2)))
end

end
