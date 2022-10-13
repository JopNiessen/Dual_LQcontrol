# Dual_LQcontrol

This algorithms computes the optimal control law numerically for a one dimentional linear quadratic system. Described as,

(1) x(t+1) = x(t) + b*u + dw

The cost to go is described by the recursive equation:

(2) J_t (x(t), theta(t)) = Ru^2 + sum_b p(b|theta) int_xi dxi N(xi|0,v) (Gx(t+1)^2 + J_{t+1} (x(t+1), theta(t+1)))

And in the final state

(3) J_T = Fx(T)^2

This equation solved by numerical integration over both bias term (b) and noise (xi) in range b in [-1, 1] and xi in [-10 sqrt(v), 10 sqrt(v)].

The optimal control (u*) is found by gradient descent, that is minimizing J over u (dJ/du = 0). 

Cost function (2) is iterated backward in time (dynamic programming) to compute the optimal cost-to-go over discrete states [linspace(-9,9,20)], beliefs [polynomialspace(-13,13,50)] and time [linspace(0,30,31)].


## Still to do\\
This version does not include the following\\
A. *Conjugate* gradient descent\\
B. Cubic spline interpolation between the (x,theta) grid points.

