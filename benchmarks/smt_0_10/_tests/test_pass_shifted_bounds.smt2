; shifted bounds on x -> both bounds
(set-logic QF_LIA)
(declare-fun x () Int)
(assert (and (>= (+ x 2) 0) (<= (+ x 2) 10)))
(assert (<= 0 x))
(assert (<= x 10))
(check-sat)


