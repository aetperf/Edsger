""" Module dedicated to the generation of small networks, for testing purposes.
"""

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
       tail  head   trav_time
    0     1     2  6.000816
    1     1     3  4.008691
    2     2     1  6.000834
    3     2     6  6.573598
    4     3     1  4.008587

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
            1,2,6.0008162373543197 
            1,3,4.0086907502079407 
            2,1,6.0008341229953821 
            2,6,6.5735982553868011 
            3,1,4.0085866534998482 
            3,4,4.2694018322732905 
            3,12,4.0201791556206405 
            4,3,4.2712677558757353 
            4,5,2.3153741062577953 
            4,11,7.1333004801798925 
            5,4,2.3170722283501695 
            5,6,9.9982252077098899 
            5,9,9.651310705325999 
            6,2,6.5995176673701739 
            6,5,10.020702556347622 
            6,8,14.690955002063726 
            7,8,5.5521603811029898 
            7,18,2.0622256872131777 
            8,6,14.824159517828813 
            8,7,5.5014129625630854 
            8,9,15.174707514675859 
            8,16,10.729473525552692 
            9,5,9.6701545595005562 
            9,8,15.03786950444767 
            9,10,5.6825330516020252 
            10,9,5.717243386296829 
            10,11,12.405689451182845 
            10,15,13.722370282505469 
            10,16,20.084809978398383 
            10,17,16.308017150740422 
            11,4,7.2230245551941348 
            11,10,12.203254534036494 
            11,12,13.590227634461069 
            11,14,13.691285688495954 
            12,3,4.019790487445956 
            12,11,13.735155648868583 
            12,13,3.0227965436823721 
            13,12,3.0234796715615868 
            13,24,17.661007722734873 
            14,11,13.842645045035516 
            14,15,12.23433912804607 
            14,23,9.0793443117183674 
            15,10,13.811560451025963 
            15,14,12.374604857173303 
            15,19,4.3262120793321053 
            15,22,9.0881436730596299 
            16,8,10.778811570380915 
            16,10,20.236275698759833 
            16,17,9.5014584909994753 
            16,18,3.1634648042599296 
            17,10,16.308017150740422 
            17,16,9.472854415655231 
            17,19,7.436626799094368 
            18,7,2.0631863850180094 
            18,16,3.1658348757764214 
            18,20,4.2593710873324016 
            19,15,4.335530792063869 
            19,17,7.4122740353295606 
            19,20,9.4590635081531964 
            20,18,4.2602300670551969 
            20,19,9.5152493985015738 
            20,21,8.1656608127813008 
            20,22,7.7131300003052283 
            21,20,8.0816356896656067 
            21,22,4.2135058260885172 
            21,24,11.924059828422909 
            22,15,9.0571671827464755 
            22,20,7.7131300003052283 
            22,21,4.2010498558110108 
            22,23,12.365805495832021 
            23,14,9.0659665440877273 
            23,22,12.243138489387317 
            23,24,3.7593041884018925 
            24,13,17.617020723058587 
            24,21,11.752579405401582 
            24,23,3.7229467421027662"""
        )
        graph_edges = pd.read_csv(
            siouxfalls_graph_edges_csv,
            sep=",",
            dtype={"tail": np.int32, "head": np.int32, "trav_time": np.float64},
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
