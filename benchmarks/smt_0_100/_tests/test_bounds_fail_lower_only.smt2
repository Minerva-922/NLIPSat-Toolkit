; only lower bound -> should be only_bounds_failed
(set-logic QF_LIA)
(declare-fun x () Int)
(assert (>= x 0))
(assert (<= x 100))
(check-sat)


