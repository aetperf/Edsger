""" Module dedicated to the generation of small networks, for testing purposes.
"""

import numpy as np
import pandas as pd


def create_Spiess_network(
    dwell_time=5.0, a_very_small_time_interval=1.0e-06, board_alight_ratio=0.5
):
    """Example network from Heinz Spiess, Michael Florian,
    Optimal strategies: A new assignment model for transit networks,
    Transportation Research Part B: Methodological,
    Volume 23, Issue 2, 1989, Pages 83-102, ISSN 0191-2615,
    https://doi.org/10.1016/0191-2615(89)90034-9.

    This network has 13 vertices and 24 edges.
    """

    boarding_time = board_alight_ratio * dwell_time
    alighting_time = board_alight_ratio * dwell_time + a_very_small_time_interval

    line1_freq = 1.0 / (60.0 * 12.0)
    line2_freq = 1.0 / (60.0 * 12.0)
    line3_freq = 1.0 / (60.0 * 30.0)
    line4_freq = 1.0 / (60.0 * 6.0)

    # stop A
    # 0 stop vertex
    # 1 boarding vertex
    # 2 boarding vertex

    # stop X
    # 3 stop vertex
    # 4 boarding vertex
    # 5 alighting vertex
    # 6 boarding vertex

    # stop Y
    # 7  stop vertex
    # 8  boarding vertex
    # 9  alighting vertex
    # 10 boarding vertex
    # 11 alighting vertex

    # stop B
    # 12 stop vertex
    # 13 alighting vertex
    # 14 alighting vertex
    # 15 alighting vertex

    tail = []
    head = []
    trav_time = []
    freq = []

    # edge 0
    # stop A : to line 1
    # boarding edge
    tail.append(0)
    head.append(2)
    freq.append(line1_freq)
    trav_time.append(boarding_time)

    # edge 1
    # stop A : to line 2
    # boarding edge
    tail.append(0)
    head.append(1)
    freq.append(line2_freq)
    trav_time.append(boarding_time)

    # edge 2
    # line 1 : first segment
    # on-board edge
    tail.append(2)
    head.append(15)
    freq.append(np.inf)
    trav_time.append(25.0 * 60.0)

    # edge 3
    # line 2 : first segment
    # on-board edge
    tail.append(1)
    head.append(5)
    freq.append(np.inf)
    trav_time.append(7.0 * 60.0)

    # edge 4
    # stop X : from line 2
    # alighting edge
    tail.append(5)
    head.append(3)
    freq.append(np.inf)
    trav_time.append(alighting_time)

    # edge 5
    # stop X : in line 2
    # dwell edge
    tail.append(5)
    head.append(6)
    freq.append(np.inf)
    trav_time.append(dwell_time)

    # edge 6
    # stop X : from line 2 to line 3
    # transfer edge
    tail.append(5)
    head.append(4)
    freq.append(line3_freq)
    trav_time.append(dwell_time)

    # edge 7
    # stop X : to line 2
    # boarding edge
    tail.append(3)
    head.append(6)
    freq.append(line2_freq)
    trav_time.append(boarding_time)

    # edge 8
    # stop X : to line 3
    # boarding edge
    tail.append(3)
    head.append(4)
    freq.append(line3_freq)
    trav_time.append(boarding_time)

    # edge 9
    # line 2 : second segment
    # on-board edge
    tail.append(6)
    head.append(11)
    freq.append(np.inf)
    trav_time.append(6.0 * 60.0)

    # edge 10
    # line 3 : first segment
    # on-board edge
    tail.append(4)
    head.append(9)
    freq.append(np.inf)
    trav_time.append(4.0 * 60.0)

    # edge 11
    # stop Y : from line 3
    # alighting edge
    tail.append(9)
    head.append(7)
    freq.append(np.inf)
    trav_time.append(alighting_time)

    # edge 12
    # stop Y : from line 2
    # alighting edge
    tail.append(11)
    head.append(7)
    freq.append(np.inf)
    trav_time.append(alighting_time)

    # edge 13
    # stop Y : from line 2 to line 3
    # transfer edge
    tail.append(11)
    head.append(10)
    freq.append(line3_freq)
    trav_time.append(dwell_time)

    # edge 14
    # stop Y : from line 2 to line 4
    # transfer edge
    tail.append(11)
    head.append(8)
    freq.append(line4_freq)
    trav_time.append(dwell_time)

    # edge 15
    # stop Y : from line 3 to line 4
    # transfer edge
    tail.append(9)
    head.append(8)
    freq.append(line4_freq)
    trav_time.append(dwell_time)

    # edge 16
    # stop Y : in line 3
    # dwell edge
    tail.append(9)
    head.append(10)
    freq.append(np.inf)
    trav_time.append(dwell_time)

    # edge 17
    # stop Y : to line 3
    # boarding edge
    tail.append(7)
    head.append(10)
    freq.append(line3_freq)
    trav_time.append(boarding_time)

    # edge 18
    # stop Y : to line 4
    # boarding edge
    tail.append(7)
    head.append(8)
    freq.append(line4_freq)
    trav_time.append(boarding_time)

    # edge 19
    # line 3 : second segment
    # on-board edge
    tail.append(10)
    head.append(14)
    freq.append(np.inf)
    trav_time.append(4.0 * 60.0)

    # edge 20
    # line 4 : first segment
    # on-board edge
    tail.append(8)
    head.append(13)
    freq.append(np.inf)
    trav_time.append(10.0 * 60.0)

    # edge 21
    # stop Y : from line 1
    # alighting edge
    tail.append(15)
    head.append(12)
    freq.append(np.inf)
    trav_time.append(alighting_time)

    # edge 22
    # stop Y : from line 3
    # alighting edge
    tail.append(14)
    head.append(12)
    freq.append(np.inf)
    trav_time.append(alighting_time)

    # edge 23
    # stop Y : from line 4
    # alighting edge
    tail.append(13)
    head.append(12)
    freq.append(np.inf)
    trav_time.append(alighting_time)

    edges_df = pd.DataFrame(
        data={"tail": tail, "head": head, "trav_time": trav_time, "freq": freq}
    )

    return edges_df
