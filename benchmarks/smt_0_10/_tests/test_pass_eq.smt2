; x is fixed by equality
(set-logic QF_LIA)
(declare-fun x () Int)
(assert (= x 5))
(check-sat)


