"""Module dedicated to the generation of small networks, for testing purposes."""

from io import StringIO

import numpy as np
import pandas as pd


class SiouxFalls:
    """
    A class representing the Sioux Falls network.

    Attributes
    ----------
    graph_edges : pandas.DataFrame
        A DataFrame containing the edges of the Sioux Falls network. The DataFrame has three
        columns: 'tail' (int32), 'head' (int32), and 'trav_time' (float64). Each row represents
        an edge from the tail node to the head node with a corresponding travel time.

    Examples
    --------
    >>> siouxfalls = SiouxFalls()
    >>> print(siouxfalls.edges.head())
       tail  head  trav_time
    0     0     1   6.000816
    1     0     2   4.008691
    2     1     0   6.000834
    3     1     5   6.573598
    4     2     0   4.008587

    """

    @property
    def edges(self):
        """
        A DataFrame containing the edges of the Sioux Falls network.

        Returns
        -------
        graph_edges : pandas.DataFrame
            A DataFrame containing the edges of the Sioux Falls network. The DataFrame has three
            columns: 'tail' (int32), 'head' (int32), and 'trav_time' (float64). Each row represents
            an edge from the tail node to the head node with a corresponding travel time.

        """
        siouxfalls_graph_edges_csv = StringIO(
            """tail,head,trav_time
            0,1,6.00081623735432
            0,2,4.008690750207941
            1,0,6.000834122995382
            1,5,6.573598255386801
            2,0,4.008586653499848
            2,3,4.2694018322732905
            2,11,4.0201791556206405
            3,2,4.271267755875735
            3,4,2.3153741062577957
            3,10,7.1333004801798925
            4,3,2.317072228350169
            4,5,9.99822520770989
            4,8,9.651310705326
            5,1,6.599517667370174
            5,4,10.020702556347622
            5,7,14.690955002063726
            6,7,5.55216038110299
            6,17,2.0622256872131777
            7,5,14.824159517828813
            7,6,5.5014129625630845
            7,8,15.17470751467586
            7,15,10.729473525552692
            8,4,9.670154559500556
            8,7,15.03786950444767
            8,9,5.682533051602025
            9,8,5.717243386296829
            9,10,12.405689451182845
            9,14,13.722370282505468
            9,15,20.084809978398383
            9,16,16.308017150740422
            10,3,7.223024555194135
            10,9,12.203254534036494
            10,11,13.590227634461067
            10,13,13.691285688495954
            11,2,4.019790487445956
            11,10,13.735155648868584
            11,12,3.022796543682372
            12,11,3.023479671561587
            12,23,17.661007722734873
            13,10,13.842645045035516
            13,14,12.23433912804607
            13,22,9.079344311718367
            14,9,13.811560451025963
            14,13,12.374604857173304
            14,18,4.326212079332104
            14,21,9.08814367305963
            15,7,10.778811570380917
            15,9,20.236275698759837
            15,16,9.501458490999475
            15,17,3.1634648042599296
            16,9,16.308017150740422
            16,15,9.472854415655233
            16,18,7.436626799094368
            17,6,2.063186385018009
            17,15,3.1658348757764214
            17,19,4.2593710873324016
            18,14,4.335530792063869
            18,16,7.41227403532956
            18,19,9.459063508153196
            19,17,4.260230067055197
            19,18,9.515249398501572
            19,20,8.165660812781299
            19,21,7.713130000305229
            20,19,8.081635689665607
            20,21,4.213505826088517
            20,23,11.924059828422909
            21,14,9.057167182746477
            21,19,7.713130000305229
            21,20,4.201049855811011
            21,22,12.36580549583202
            22,13,9.065966544087727
            22,21,12.243138489387317
            22,23,3.759304188401893
            23,12,17.617020723058587
            23,20,11.752579405401582
            23,22,3.722946742102766"""
        )
        graph_edges = pd.read_csv(
            siouxfalls_graph_edges_csv,
            sep=",",
            dtype={"tail": np.uint32, "head": np.uint32, "trav_time": np.float64},
        )
        return graph_edges


def create_sf_network(dwell_time=1.0e-6, board_alight_ratio=0.5):
    """
    Example network from Spiess, H. and Florian, M. (1989).
    Optimal strategies: A new assignment model for transit networks.
    Transportation Research Part B 23(2), 83-102.

    This network has 13 vertices and 24 edges.
    """

    boarding_time = board_alight_ratio * dwell_time
    alighting_time = board_alight_ratio * dwell_time

    line1_freq = 1.0 / (60.0 * 12.0)
    line2_freq = 1.0 / (60.0 * 12.0)
    line3_freq = 1.0 / (60.0 * 30.0)
    line4_freq = 1.0 / (60.0 * 6.0)

    # stop A
    # 0 stop vertex
    # 1 boarding vertex64
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
    vol = []

    # edge 0
    # stop A : to line 1
    # boarding edge
    tail.append(0)
    head.append(2)
    freq.append(line1_freq)
    trav_time.append(boarding_time)
    vol.append(0.5)

    # edge 1
    # stop A : to line 2
    # boarding edge
    tail.append(0)
    head.append(1)
    freq.append(line2_freq)
    trav_time.append(boarding_time)
    vol.append(0.5)

    # edge 2
    # line 1 : first segment
    # on-board edge
    tail.append(2)
    head.append(15)
    freq.append(np.inf)
    trav_time.append(25.0 * 60.0)
    vol.append(0.5)

    # edge 3
    # line 2 : first segment
    # on-board edge
    tail.append(1)
    head.append(5)
    freq.append(np.inf)
    trav_time.append(7.0 * 60.0)
    vol.append(0.5)

    # edge 4
    # stop X : from line 2
    # alighting edge
    tail.append(5)
    head.append(3)
    freq.append(np.inf)
    trav_time.append(alighting_time)
    vol.append(0.0)

    # edge 5
    # stop X : in line 2
    # dwell edge
    tail.append(5)
    head.append(6)
    freq.append(np.inf)
    trav_time.append(dwell_time)
    vol.append(0.5)

    # edge 6
    # stop X : from line 2 to line 3
    # transfer edge
    tail.append(5)
    head.append(4)
    freq.append(line3_freq)
    trav_time.append(dwell_time)
    vol.append(0.0)

    # edge 7
    # stop X : to line 2
    # boarding edge
    tail.append(3)
    head.append(6)
    freq.append(line2_freq)
    trav_time.append(boarding_time)
    vol.append(0.0)

    # edge 8
    # stop X : to line 3
    # boarding edge
    tail.append(3)
    head.append(4)
    freq.append(line3_freq)
    trav_time.append(boarding_time)
    vol.append(0.0)

    # edge 9
    # line 2 : second segment
    # on-board edge
    tail.append(6)
    head.append(11)
    freq.append(np.inf)
    trav_time.append(6.0 * 60.0)
    vol.append(0.5)

    # edge 10
    # line 3 : first segment
    # on-board edge
    tail.append(4)
    head.append(9)
    freq.append(np.inf)
    trav_time.append(4.0 * 60.0)
    vol.append(0.0)

    # edge 11
    # stop Y : from line 3
    # alighting edge
    tail.append(9)
    head.append(7)
    freq.append(np.inf)
    trav_time.append(alighting_time)
    vol.append(0.0)

    # edge 12
    # stop Y : from line 2
    # alighting edge
    tail.append(11)
    head.append(7)
    freq.append(np.inf)
    trav_time.append(alighting_time)
    vol.append(0.0)

    # edge 13
    # stop Y : from line 2 to line 3
    # transfer edge
    tail.append(11)
    head.append(10)
    freq.append(line3_freq)
    trav_time.append(dwell_time)
    vol.append(0.0833333333333)

    # edge 14
    # stop Y : from line 2 to line 4
    # transfer edge
    tail.append(11)
    head.append(8)
    freq.append(line4_freq)
    trav_time.append(dwell_time)
    vol.append(0.4166666666666)

    # edge 15
    # stop Y : from line 3 to line 4
    # transfer edge
    tail.append(9)
    head.append(8)
    freq.append(line4_freq)
    trav_time.append(dwell_time)
    vol.append(0.0)

    # edge 16
    # stop Y : in line 3
    # dwell edge
    tail.append(9)
    head.append(10)
    freq.append(np.inf)
    trav_time.append(dwell_time)
    vol.append(0.0)

    # edge 17
    # stop Y : to line 3
    # boarding edge
    tail.append(7)
    head.append(10)
    freq.append(line3_freq)
    trav_time.append(boarding_time)
    vol.append(0.0)

    # edge 18
    # stop Y : to line 4
    # boarding edge
    tail.append(7)
    head.append(8)
    freq.append(line4_freq)
    trav_time.append(boarding_time)
    vol.append(0.0)

    # edge 19
    # line 3 : second segment
    # on-board edge
    tail.append(10)
    head.append(14)
    freq.append(np.inf)
    trav_time.append(4.0 * 60.0)
    vol.append(0.0833333333333)

    # edge 20
    # line 4 : first segment
    # on-board edge
    tail.append(8)
    head.append(13)
    freq.append(np.inf)
    trav_time.append(10.0 * 60.0)
    vol.append(0.4166666666666)

    # edge 21
    # stop Y : from line 1
    # alighting edge
    tail.append(15)
    head.append(12)
    freq.append(np.inf)
    trav_time.append(alighting_time)
    vol.append(0.5)

    # edge 22
    # stop Y : from line 3
    # alighting edge
    tail.append(14)
    head.append(12)
    freq.append(np.inf)
    trav_time.append(alighting_time)
    vol.append(0.0833333333333)

    # edge 23
    # stop Y : from line 4
    # alighting edge
    tail.append(13)
    head.append(12)
    freq.append(np.inf)
    trav_time.append(alighting_time)
    vol.append(0.4166666666666)

    edges = pd.DataFrame(
        data={
            "tail": tail,
            "head": head,
            "trav_time": trav_time,
            "freq": freq,
            "volume_ref": vol,
        }
    )
    # waiting time is in average half of the period
    edges["freq"] *= 2.0

    return edges


# author : Francois Pacull
# copyright : Architecture & Performance
# email: francois.pacull@architecture-performance.fr
# license : MIT
