import cvxpy as cp
import numpy as np

# Problem parameters (set these to your values)
alpha1, alpha2, alpha3 = 1.0, 1.0, 1.0

# Decision variable x = [x1, ..., x5]
x = cp.Variable(5)

# Objective: -a1*x1 - a2*x2 + a3*(x3 - 8)^2
# (x[0] is x1, x[1] is x2, x[2] is x3, ...)
objective = cp.Minimize(
    -alpha1 * x[0] - alpha2 * x[1] + alpha3 * cp.square(x[2] - 8)
)

constraints = [
    x[2] - 9 <= 0,            # x3 - 9 <= 0
    -x[2] + 7 <= 0,           # -x3 + 7 <= 0
    -x[4] + 2 <= 0,           # -x5 + 2 <= 0
    cp.sum(x) - 24 == 0,      # x1 + x2 + x3 + x4 + x5 - 24 = 0
    x >= 0,
]

prob = cp.Problem(objective, constraints)
prob.solve()

print("status:", prob.status)
print("optimal value:", prob.value)
print("x* =", x.value)