%% Example code to use MPC
% Edgar Bolivar - Nov. 2023
%
% Calls the CVXGEN-generated MPC solver (csolve) once. The problem it solves
% is defined in description.cvxgen (5 states, 2 inputs, horizon T = 10):
%
%   minimize   sum_{t=0..T} (x_t'*Q*x_t + u_t'*R*u_t) + x_{T+1}'*Q_final*x_{T+1}
%   subject to x_{t+1} = A*x_t + B*u_t       (dynamics)
%              |u_t| <= u_max                (actuator limit, each input)
%              ||u_{t+1} - u_t||_inf <= S    (slew-rate limit)
%
% This is a single open-loop solve on a random system, to show the solver's
% interface and speed. A real MPC controller would re-solve every time step
% from the measured state and apply only u_0 (receding horizon).
%
% Requires: run from this folder, so csolve.mexw64 is on the path. The
%           prebuilt mex is Windows-only; elsewhere run make_csolve first
%           (needs a C compiler, see "mex -setup").
% Check:    cvxsolve(params) solves the same problem with CVX.

clc, clearvars, close all

% Problem data. Sizes must match description.cvxgen.
params.A = randn(5,5);          % dynamics matrix (n x n)
params.B = randn(5,2);          % input matrix    (n x m)
params.Q = eye(5);              % state cost, PSD
params.Q_final = eye(5);        % final state cost, PSD
params.R = eye(2);              % input cost, PSD
params.S = 10;                  % slew-rate limit
params.u_max = 2;               % input amplitude limit
params.x_0 = ones(5,1);         % initial state

% Exercise the high-speed solver.
tic
[vars, status] = csolve(params);  % solve, saving results.
toc

% Check convergence, and display the optimal variable value.
% vars.x{t} and vars.u{t} hold the predicted states and inputs.
if ~status.converged, error 'failed to converge'; end
vars.x
