def part_obj_to_datasetclass(obj_classes=21, animate=True):

    map_pc = {}
    for i in range(obj_classes):
        map_pc[i] = {}
        
    # Animate objects

    map_pc[3][1] = 8  # Bird
    map_pc[3][5] = 9
    map_pc[3][3] = 10
    map_pc[3][2] = 11

    map_pc[8][1] = 23  # Cat
    map_pc[8][3] = 24
    map_pc[8][4] = 25
    map_pc[8][2] = 26

    map_pc[10][1] = 28  # Cow
    map_pc[10][4] = 29
    map_pc[10][3] = 30
    map_pc[10][2] = 31

    map_pc[12][1] = 33  # Dog
    map_pc[12][3] = 34
    map_pc[12][4] = 35
    map_pc[12][2] = 36

    map_pc[13][1] = 37  # Horse
    map_pc[13][4] = 38
    map_pc[13][3] = 39
    map_pc[13][2] = 40

    map_pc[15][1] = 43  # Person
    map_pc[15][2] = 44
    map_pc[15][7] = 45
    map_pc[15][6] = 46
    map_pc[15][8] = 47
    map_pc[15][3] = 48

    map_pc[17][1] = 51  # Sheep
    map_pc[17][3] = 52
    map_pc[17][2] = 53

    
    # Inanimate objects
    map_pc[1][1] = 1  # Aeroplane
    map_pc[1][5] = 2
    map_pc[1][3] = 3
    map_pc[1][4] = 4
    map_pc[1][2] = 5
    
    map_pc[2][2] = 6 # Bicycle
    map_pc[2][1] = 7
    
    map_pc[4][0] = 12  # Boat

    map_pc[5][13] = 13 # Bottle
    map_pc[5][14] = 14

    map_pc[6][12] = 15 # Bus
    map_pc[6][2] = 16
    map_pc[6][1] = 17

    map_pc[7][12] = 18 # Car
    map_pc[7][2] = 19
    map_pc[7][6] = 20
    map_pc[7][7] = 21
    map_pc[7][1] = 22
    
    map_pc[9][0] = 27  # Chair
    
    map_pc[11][0] = 32  # Dining Table

    map_pc[14][2] = 41  # Motorbike
    map_pc[14][1] = 42

    map_pc[16][10] = 49  # Potted plant
    map_pc[16][11] = 50
    
    map_pc[18][0] = 54  # Sofa

    map_pc[19][0] = 55  # Train

    map_pc[20][8] = 56  # Tv monitor
    map_pc[20][9] = 57
    
    if animate is None:
        classes = list(range(1, 21))
    elif animate:
        classes = [3, 8, 10, 12, 13, 15, 17]
    else:
        classes = [1, 2, 4, 5, 6, 7, 9, 11, 14, 16, 18, 19, 20]

    return map_pc, classes


def aggregate_parts_to_classes(num_classes=58, level=1, animate=True):
    
    # Level 1 animate parts : Head (1), Torso (2), (Upper) Leg (3), Tail (4), Wing(5),
    #                         Upper Arm (6), Lower Arm(7), Lower Leg(8)

    map_pc = {}
    for i in range(num_classes):
        map_pc[i] = 0

    if animate:
        map_pc[8] = 1  # Bird
        map_pc[9] = 5
        map_pc[10] = 3
        map_pc[11] = 2

        map_pc[23] = 1  # Cat
        map_pc[24] = 3
        map_pc[25] = 4
        map_pc[26] = 2

        map_pc[28] = 1  # Cow
        map_pc[29] = 4
        map_pc[30] = 3
        map_pc[31] = 2

        map_pc[33] = 1  # Dog
        map_pc[34] = 3
        map_pc[35] = 4
        map_pc[36] = 2

        map_pc[37] = 1  # Horse
        map_pc[38] = 4
        map_pc[39] = 3
        map_pc[40] = 2

        map_pc[43] = 1  # Person
        map_pc[44] = 2
        map_pc[45] = 7
        map_pc[46] = 6
        map_pc[47] = 8
        map_pc[48] = 3

        map_pc[51] = 1  # Sheep
        map_pc[52] = 3
        map_pc[53] = 2
        
    # Level 1 inanimate parts : Body (1), Wheel (2), Wing (3), Stern (4), Engine(5), Light (6)
    #                           Plate (7), Screen (8), Frame (9), Pot (10), Plant (11), Window (12),
    #                           Bottle Cap (13), Bottle Body (14)

    else:
        map_pc[1] = 1  # Aeroplane
        map_pc[2] = 5
        map_pc[3] = 3
        map_pc[4] = 4
        map_pc[5] = 2

        map_pc[6] = 2 # Bicycle
        map_pc[7] = 1

        map_pc[13] = 13 # Bottle
        map_pc[14] = 14

        map_pc[15] = 12 # Bus
        map_pc[16] = 2
        map_pc[17] = 1

        map_pc[18] = 12  # Car
        map_pc[19] = 2
        map_pc[20] = 6
        map_pc[21] = 7
        map_pc[22] = 1

        map_pc[41] = 2 # Motorbike
        map_pc[42] = 1

        map_pc[49] = 10 # Potted plant
        map_pc[50] = 11

        map_pc[56] = 8 # Tv monitor
        map_pc[57] = 9

    return map_pc
