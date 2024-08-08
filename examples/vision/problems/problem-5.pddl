(define (problem vision-5)
    (:domain vision)
    (:objects gem1 gem2 gem3 - gem
              robot1 robot2 robot3 robot4 - agent
    )
    (:init
        (= (xloc gem1) 3)
        (= (yloc gem1) 13)
        (= (xloc gem2) 7)
        (= (yloc gem2) 7)
        (= (xloc gem3) 12)
        (= (yloc gem3) 2)
        (= (walls) 
            (transpose (bit-mat 
                (bit-vec 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
                (bit-vec 0 1 1 1 0 0 1 1 1 0 1 1 1 1 0)
                (bit-vec 0 1 0 0 0 1 0 0 1 0 0 0 0 1 0)
                (bit-vec 0 1 0 1 1 1 1 0 1 0 1 1 0 1 0)
                (bit-vec 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0)
                (bit-vec 0 0 1 1 1 0 1 1 1 0 1 0 1 1 0)
                (bit-vec 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0)
                (bit-vec 0 1 0 1 1 1 0 1 0 1 1 1 0 1 0)
                (bit-vec 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0)
                (bit-vec 0 0 1 1 1 0 1 1 1 0 1 1 1 1 0)
                (bit-vec 0 1 0 0 1 0 1 0 0 0 1 0 0 0 0)
                (bit-vec 0 1 0 1 1 0 1 0 1 1 1 0 1 1 0)
                (bit-vec 0 1 0 0 0 0 1 0 0 0 0 0 0 1 0)
                (bit-vec 0 1 1 1 1 0 1 1 1 0 1 1 1 1 0)
                (bit-vec 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)))
        )
        (= (xloc robot1) 1)
        (= (yloc robot1) 1)
        (= (xloc robot2) 13)
        (= (yloc robot2) 13)
        (= (xloc robot3) 1)
        (= (yloc robot3) 13)
        (= (xloc robot4) 13)
        (= (yloc robot4) 1)
    )
    (:goal (and (offgrid gem1) (offgrid gem2) (offgrid gem3))) ; all gems are off the grid (collected)
)