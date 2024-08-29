(define (problem simple)
    (:domain vision)
    (:objects red_gem1 - red
              blue_gem2 - blue
              robot1 - agent
    )
    (:init
        (= (xloc red_gem1) 2)
        (= (yloc red_gem1) 2)
        (= (xloc blue_gem2) 4)
        (= (yloc blue_gem2) 2)
        (= (walls) 
            (transpose (bit-mat 
                (bit-vec 1 0 1 0)
                (bit-vec 1 0 1 0)
                (bit-vec 0 0 0 0)
                (bit-vec 1 1 0 1)))
        )
        (= (xloc robot1) 3)
        (= (yloc robot1) 4)
    )
    (:goal (and (has robot1 red_gem1) (has robot1 blue_gem2)))
)
