(define (domain vision)
    (:requirements :fluents :adl :typing)
    (:types carrot onion tomato - item - object)
    (:predicates 
        (has ?i - item)
        (offgrid ?i - item)
        (visible ?i - item)
    )
    (:functions 
        (xpos) (ypos) - integer
        (xloc ?o - object) (yloc ?o - object) - integer
        (walls)- bit-matrix
    )
    (:action pickup
        :parameters (?i - item)
        :precondition (and (not (has ?i)) (= xpos (xloc ?i)) (= ypos (yloc ?i)))
        :effect (and (has ?i) (offgrid ?i)
                     (assign (xloc ?i) -10) (assign (yloc ?i) -10))
    )
    (:action up
        :precondition (and (> ypos 1)
                        (= (get-index walls (- ypos 1) xpos) false))
        :effect (and (decrease ypos 1)
                    (forall (?i - item) ; check up
                        (when (or (and (= (xloc ?i) xpos) 
                                        (= (yloc ?i) (- ypos 2)))
                                (and (= (xloc ?i) xpos) 
                                        (= (yloc ?i) (- ypos 3))
                                        (= (get-index walls (- ypos 2) xpos) false))
                                (and (= (xloc ?i) xpos) 
                                        (= (yloc ?i) (- ypos 4))
                                        (or (= (get-index walls (- ypos 2) xpos) false)
                                            (= (get-index walls (- ypos 3) xpos) false))))
                            (visible ?i)))
                    (forall (?i - item) ; check right
                        (when (or (and (= (xloc ?i) (+ xpos 1)) 
                                        (= (yloc ?i) (- ypos 1)))
                                (and (= (xloc ?i) (+ xpos 2)) 
                                        (= (yloc ?i) (- ypos 1))
                                        (= (get-index walls (- ypos 1) (+ xpos 1)) false))
                                (and (= (xloc ?i) (+ xpos 3)) 
                                        (= (yloc ?i) (- ypos 1))
                                        (or (= (get-index walls (- ypos 1) (+ xpos 1)) false)
                                            (= (get-index walls (- ypos 1) (+ xpos 2)) false))))
                            (visible ?i)))
                    (forall (?i - item) ; check left
                        (when (or (and (= (xloc ?i) (- xpos 1)) 
                                        (= (yloc ?i) (- ypos 1)))
                                (and (= (xloc ?i) (- xpos 2)) 
                                        (= (yloc ?i) (- ypos 1))
                                        (= (get-index walls (- ypos 1) (- xpos 1)) false))
                                (and (= (xloc ?i) (- xpos 3)) 
                                        (= (yloc ?i) (- ypos 1))
                                        (or (= (get-index walls (- ypos 1) (- xpos 1)) false)
                                            (= (get-index walls (- ypos 1) (- xpos 2)) false))))
                            (visible ?i)))))
    (:action down
        :precondition (and (< ypos (height walls))
                        (= (get-index walls (+ ypos 1) xpos) false))
        :effect (and 
                    (increase ypos 1)
                    (forall (?i - item) ; check down
                        (when (or (and (= (xloc ?i) xpos) 
                                        (= (yloc ?i) (+ ypos 2)))
                                (and (= (xloc ?i) xpos) 
                                        (= (yloc ?i) (+ ypos 3))
                                        (= (get-index walls (+ ypos 2) xpos) false))
                                (and (= (xloc ?i) xpos) 
                                        (= (yloc ?i) (+ ypos 4))
                                        (or (= (get-index walls (+ ypos 2) xpos) false)
                                            (= (get-index walls (+ ypos 3) xpos) false))))
                            (visible ?i)))
                    (forall (?i - item) ; check right
                        (when (or (and (= (xloc ?i) (+ xpos 1)) 
                                        (= (yloc ?i) (+ ypos 1)))
                                (and (= (xloc ?i) (+ xpos 2)) 
                                        (= (yloc ?i) (+ ypos 1))
                                        (= (get-index walls (+ ypos 1) (+ xpos 1)) false))
                                (and (= (xloc ?i) (+ xpos 3)) 
                                        (= (yloc ?i) (+ ypos 1))
                                        (or (= (get-index walls (+ ypos 1) (+ xpos 1)) false)
                                            (= (get-index walls (+ ypos 1) (+ xpos 2)) false))))
                            (visible ?i)))
                    (forall (?i - item) ; check left
                        (when (or (and (= (xloc ?i) (- xpos 1)) 
                                        (= (yloc ?i) (+ ypos 1)))
                                (and (= (xloc ?i) (- xpos 2)) 
                                        (= (yloc ?i) (+ ypos 1))
                                        (= (get-index walls (+ ypos 1) (- xpos 1)) false))
                                (and (= (xloc ?i) (- xpos 3)) 
                                        (= (yloc ?i) ypos)
                                        (or (= (get-index walls (+ ypos 1) (- xpos 1)) false)
                                            (= (get-index walls (+ ypos 1) (- xpos 2)) false))))
                            (visible ?i)))))

    (:action left
        :precondition (and (> xpos 1)
                        (= (get-index walls ypos (- xpos 1)) false))
        :effect (and (decrease xpos 1)
                    (forall (?i - item) ; check down
                        (when (or (and (= (xloc ?i) (- xpos 1)) 
                                        (= (yloc ?i) (+ ypos 1)))
                                (and (= (xloc ?i) (- xpos 1)) 
                                        (= (yloc ?i) (+ ypos 2))
                                        (= (get-index walls (+ ypos 1) (- xpos 1)) false))
                                (and (= (xloc ?i) (- xpos 1)) 
                                        (= (yloc ?i) (+ ypos 3))
                                        (or (= (get-index walls (+ ypos 1) (- xpos 1)) false)
                                            (= (get-index walls (+ ypos 2) (- xpos 1)) false))))
                            (visible ?i)))
                    (forall (?i - item) ;check up
                        (when (or (and (= (xloc ?i) (- xpos 1)) 
                                        (= (yloc ?i) (- ypos 1)))
                                (and (= (xloc ?i) (- xpos 1)) 
                                        (= (yloc ?i) (- ypos 2))
                                        (= (get-index walls (- ypos 1) (- xpos 1)) false))
                                (and (= (xloc ?i) xpos) 
                                        (= (yloc ?i) (- ypos 3))
                                        (or (= (get-index walls (- ypos 1) (- xpos 1)) false)
                                            (= (get-index walls (- ypos 2) (- xpos 1)) false))))
                            (visible ?i)))
                    (forall (?i - item) ; check left
                        (when (or (and (= (xloc ?i) (- xpos 2)) 
                                        (= (yloc ?i) ypos))
                                (and (= (xloc ?i) (- xpos 3)) 
                                        (= (yloc ?i) ypos)
                                        (= (get-index walls ypos (- xpos 2)) false))
                                (and (= (xloc ?i) (- xpos 4)) 
                                        (= (yloc ?i) ypos)
                                        (or (= (get-index walls ypos (- xpos 2)) false)
                                            (= (get-index walls ypos (- xpos 3)) false))))
                            (visible ?i)))))

    (:action right
        :precondition (and (< xpos (width walls))
                           (= (get-index walls ypos (+ xpos 1)) false))
        :effect (and (increase xpos 1)
                    (forall (?i - item) ; check down
                        (when (or (and (= (xloc ?i) (+ xpos 1))
                                        (= (yloc ?i) (+ ypos 1)))
                                (and (= (xloc ?i) (+ xpos 1)) 
                                        (= (yloc ?i) (+ ypos 2))
                                        (= (get-index walls (+ ypos 1) (+ xpos 1)) false))
                                (and (= (xloc ?i) (+ xpos 1)) 
                                        (= (yloc ?i) (+ ypos 3))
                                        (or (= (get-index walls (+ ypos 1) (+ xpos 1)) false)
                                            (= (get-index walls (+ ypos 2) (+ xpos 1)) false))))
                            (visible ?i)))
                    (forall (?i - item) ; check up
                        (when (or (and (= (xloc ?i) (+ xpos 1)) 
                                        (= (yloc ?i) (- ypos 1)))
                                (and (= (xloc ?i) (+ xpos 1)) 
                                        (= (yloc ?i) (- ypos 2))
                                        (= (get-index walls (- ypos 1) (+ xpos 1)) false))
                                (and (= (xloc ?i) (+ xpos 1)) 
                                        (= (yloc ?i) (- ypos 3))
                                        (or (= (get-index walls (- ypos 1) (+ xpos 1)) false)
                                            (= (get-index walls (- ypos 2) (+ xpos 1)) false))))
                            (visible ?i)))
                    (forall (?i - item) ; check right
                        (when (or (and (= (xloc ?i) (+ xpos 2)) 
                                        (= (yloc ?i) ypos))
                                (and (= (xloc ?i) (+ xpos 3)) 
                                        (= (yloc ?i) ypos)
                                        (= (get-index walls ypos (+ xpos 2)) false))
                                (and (= (xloc ?i) (+ xpos 4)) 
                                        (= (yloc ?i) ypos)
                                        (or (= (get-index walls ypos (+ xpos 2)) false)
                                            (= (get-index walls ypos (+ xpos 3)) false))))
                            (visible ?i)))
        ))
)
