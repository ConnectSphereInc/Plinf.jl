(define (problem vision-2)
    (:domain vision)
    (:objects
        carrot1 - carrot 
        onion1 - onion)
    (:init
        (= (xloc carrot1) 2)
        (= (yloc carrot1) 4)
        (= (xloc onion1) 8)
        (= (yloc onion1) 4)
        (= (walls) 
            (transpose (bit-mat 
                (bit-vec 0 0 0 1 0 1 0 0 0 )
                (bit-vec 0 0 0 1 0 1 0 0 0 )
                (bit-vec 1 1 1 1 0 1 1 1 1 )
                (bit-vec 1 0 0 0 0 0 0 0 1 )
                (bit-vec 1 1 1 1 1 1 1 1 1 )))
        )
        (= (xpos) 5)
        (= (ypos) 1)
    )
    (:goal (has onion1))
)
