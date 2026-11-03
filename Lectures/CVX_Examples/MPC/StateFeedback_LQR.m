%% State-space position control of a DC motor - LQR state feedback
% Edgar Bolivar
%
% Builds the state-space model of a brushed DC motor from its datasheet,
% checks controllability, designs a state-feedback gain with LQR, and
% simulates the closed loop tracking a constant position reference r.
%
%   State  x = [theta; omega; i]  (angle [rad], speed [rad/s], current [A])
%   Input  u = winding voltage [V]
%   Output y = theta
%
%   LQR picks K in u = -K*(x - x_ref) to minimize
%   J = integral( e'*Q*e + u'*R*u ) dt,  e = x - x_ref.
%
% Requires: Control System Toolbox (lqr).
% Output:   "Controller Response" figure (position vs. reference, voltage).
% Try:      raise Q(1,1) for a faster response, raise R to use less voltage.
%           Compare the peak voltage to what the motor driver can supply.
%           This is the gap that MPC (see cvxgen/) closes by adding |u| <= u_max.

clc, clearvars, close all

%Define system parameters from datasheet. For example, PN 310005 - Maxon
Rw = 0.196;                     %Ohm            winding resistance
Lw = 0.0345e-3;                 %Henry          winding inductance
Kt = 13.9e-3;                   %N.m/amp        torque constant
Kv = Kt;                        %V/(rad/s)      back-EMF constant (= Kt in SI)
Jm = 33.7*(1/1e3)*(1/100^2);    %Kg m^2         rotor inertia (33.7 g cm^2)
bm = 3.61e-3;                   %N.m/rad*sec    viscous friction

% Create state space model
%   theta' = omega
%   Jm*omega' = -bm*omega + Kt*i          (mechanical)
%   Lw*i'     = -Kv*omega - Rw*i + u      (electrical)
A  = [  0,1,0;
    0, -bm/Jm, Kt/Jm;
    0, -Kv/Lw, -Rw/Lw];
B = [0; 0; 1/Lw];
C = [1, 0, 0];
D = 0;

% Reference position [rad]
r = 20;

%% Properties of the system

%Controllability Matrix ([B AB A^2B] is enough for 3 states; the extra
%A^3B block does not change the rank)
W = [B A*B A*A*B A*A*A*B];

% Is the system controllable?
if rank(W) == 3, disp("The system is controllable");
else, disp("The system is not controllable"); end

%% State feedback (LQR) controller design

Q = [10, 0, 0;      % weight on position error
     0, 1, 0;       % weight on speed
     0, 0, 1];      % weight on current
R = 100;            % weight on voltage

[K,S,P] = lqr(A,B,Q,R);     % K: gain, S: Riccati solution, P: closed-loop poles

%% Evaluate performance through simulation ODE45

% Simulate the closed loop on the nominal model
simulate_and_plot_ABCD(A, B, C, K, r);

function simulate_and_plot_ABCD(A, B, C, K, r)
% Simulate x' = A*e + B*u with u = -K*e, e = x - [r;0;0]. Writing A*e
% instead of A*x is valid because A's first column is zero: any constant
% angle with zero speed and current is an equilibrium.
x0      = [10, 2, 3];  % Initial conditions for the state
tspan   = [0, 40];      % Time window for simulation
[t, x]  = ode45(@(t, x) A*(x-[r;0;0]) + B*(-K*(x-[r;0;0])), tspan, x0);
x       = x.';
y       = zeros(length(x), 1);
u       = zeros(length(x), 1);
% Compute output and control input from the simulated state
for i = 1:length(x),  y(i) = C*x(:,i); u(i) = -K*(x(:,i)-[r;0;0]); end

% Plot the analysis
figure, sgtitle("Controller Response"), subplot(2,1,1), hold on, grid on
xlabel("Time [s]"), ylabel("Motor Angular Position [rad]"), title("Output")
plot(t, y)
plot(t, ones(length(x),1)*r, "--r")
legend("Output", "Reference")
subplot(2,1,2), hold on, grid on, title("Control Input")
xlabel("Time [s]"), ylabel("Voltage Input [Volts]")
plot(t, u)
end
