(define (problem two-agent)
    (:domain vision)
    (:objects tier1_gem1 - tier1_gem
              tier2_gem2 - tier2_gem
              tier3_gem3 - tier3_gem
              tier4_gem4 - tier4_gem
              robot1 - agent
    )
    (:init
        (= (xloc tier1_gem1) 2)
        (= (yloc tier1_gem1) 1)
        (= (xloc tier2_gem2) 1)
        (= (yloc tier2_gem2) 2)
        (= (xloc tier3_gem3) 3)
        (= (yloc tier3_gem3) 2)
        (= (xloc tier4_gem4) 2)
        (= (yloc tier4_gem4) 3)
        (= (walls) 
            (transpose (bit-mat 
                (bit-vec 1 0 1 )
                (bit-vec 0 0 0 )
                (bit-vec 1 0 1 )))
        )
        (= (xloc robot1) 2)
        (= (yloc robot1) 2)
    )
    (:goal (or (has robot1 tier1_gem1) (has robot1 tier2_gem2) (has robot1 tier3_gem3) (has robot1 tier4_gem4)))
)