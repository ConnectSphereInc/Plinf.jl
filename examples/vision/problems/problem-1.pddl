(define (problem vision-1)
    (:domain vision)
    (:objects
        carrot1 - carrot onion1 - onion)
    (:init
        (= (xloc carrot1) 4)
        (= (yloc carrot1) 4)
        (= (xloc onion1) 3)
        (= (yloc onion1) 3)
        (= (walls) 
            (transpose (bit-mat 
                (bit-vec 0 0 0 0)
                (bit-vec 0 1 1 0)
                (bit-vec 0 0 0 0)
                (bit-vec 0 0 0 0)))
        )
        (= (xpos) 1)
        (= (ypos) 1)
    )
    (:goal (has carrot1))
)
