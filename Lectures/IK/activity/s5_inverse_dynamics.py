"""s5 -- inverse dynamics   [NEXT WEEK -- placeholder]

Planned follow-on activity: joint torques from the same trials, two ways.

  OpenSim track
    - use results/Model_ik.osim (add back segment inertias; keep muscles off)
    - InverseDynamicsTool + <trial>_extloads.xml (already written by s0)
      -> joint moments .sto

  Barehand track
    - 2-D recursive Newton-Euler on the same 4-segment chain
    - segment mass / COM / inertia from a table (de Leva 1996, scaled by the
      subject's 60 kg / height) -- see lib/anthropometry.py (to be written)
    - joint angles from s3, differentiated twice (filtered) for the accelerations
    - measured foot GRF (results/<trial>_grf.mot) as the distal boundary condition
      -> ankle / knee / hip / lumbar net moments

  Compare, then relate peak torque and power to sit-to-stand exoskeleton
  actuator specifications.

Nothing to run yet.
"""

# `__doc__` is the triple-quoted string above.  `raise SystemExit(msg)` prints
# `msg` and stops the program -- so running this file just shows the plan.
raise SystemExit(__doc__)
