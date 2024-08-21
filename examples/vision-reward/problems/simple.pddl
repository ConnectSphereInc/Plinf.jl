(define (problem simple)
    (:domain vision)
    (:objects gem1 - red
              gem2 - blue
              gem3 - yellow
              gem4 - green
              robot1 - agent
    )
    (:init
        (= (xloc gem1) 2)
        (= (yloc gem1) 1)
        (= (xloc gem2) 1)
        (= (yloc gem2) 2)
        (= (xloc gem3) 3)
        (= (yloc gem3) 2)
        (= (xloc gem4) 2)
        (= (yloc gem4) 3)
        (= (walls) 
            (transpose (bit-mat 
                (bit-vec 1 0 1 )
                (bit-vec 0 0 0 )
                (bit-vec 1 0 1 )))
        )
        (= (xloc robot1) 2)
        (= (yloc robot1) 2)
        (visible robot1 gem1)
        (visible robot1 gem2)
        (visible robot1 gem3)
        (visible robot1 gem4)
    )
    (:goal (or (has robot1 gem1) (has robot1 gem2) (has robot1 gem3) (has robot1 gem4)))
)