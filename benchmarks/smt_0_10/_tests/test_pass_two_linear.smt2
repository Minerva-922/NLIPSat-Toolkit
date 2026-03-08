; two single-variable linear inequalities imply both bounds
(set-logic QF_LIA)
(declare-fun x () Int)
; 3x + 5 <= 20  -> x <= 5
(assert (<= (+ (* 3 x) 5) 20))
; x - 2 >= 0    -> x >= 2
(assert (>= (+ x (- 2)) 0))
(assert (<= 0 x))
(assert (<= x 10))
(check-sat)


