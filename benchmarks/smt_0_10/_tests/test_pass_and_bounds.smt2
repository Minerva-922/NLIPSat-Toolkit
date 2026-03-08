; x has explicit lower and upper bounds via and
(set-logic QF_LIA)
(declare-fun x () Int)
(assert (and (>= x 0) (<= x 10)))
(check-sat)


