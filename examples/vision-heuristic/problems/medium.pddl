(define (problem medium)
    (:domain vision)
    (:objects
        red_gem1 red_gem2 - red
        blue_gem1 blue_gem2 - blue
        yellow_gem1 - yellow
        green_gem1 green_gem2 - green
        robot1 - agent
    )
    (:init
        (= (xloc red_gem1) 6)
        (= (yloc red_gem1) 1)
        (= (xloc red_gem2) 5)
        (= (yloc red_gem2) 5)
        (= (xloc blue_gem1) 6)
        (= (yloc blue_gem1) 2)
        (= (xloc blue_gem2) 2)
        (= (yloc blue_gem2) 8)
        (= (xloc yellow_gem1) 5)
        (= (yloc yellow_gem1) 8)
        (= (xloc green_gem1) 7)
        (= (yloc green_gem1) 3)
        (= (xloc green_gem2) 6)
        (= (yloc green_gem2) 7)
        (= (walls) 
            (transpose (bit-mat 
                (bit-vec 0 0 0 0 0 0 0 0 0 0)
                (bit-vec 0 1 1 1 0 0 1 1 1 0)
                (bit-vec 0 1 0 0 0 1 0 0 1 0)
                (bit-vec 0 1 0 1 0 1 1 0 1 0)
                (bit-vec 0 0 0 1 0 0 0 0 0 0)
                (bit-vec 0 1 1 0 0 1 1 1 0 1)
                (bit-vec 0 1 0 0 1 0 0 1 0 1)
                (bit-vec 0 0 0 1 0 1 0 0 0 0)
                (bit-vec 1 1 0 0 0 0 1 1 0 1)
                (bit-vec 1 1 1 0 0 0 0 0 0 1)))
        )
        (= (xloc robot1) 1)
        (= (yloc robot1) 1)
        ; (= (xloc robot2) 5)
        ; (= (yloc robot2) 10)
    )
    (:goal (or (has robot1 yellow_gem1) (has robot1 yellow_gem1)))
)