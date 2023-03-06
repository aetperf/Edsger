
.. image:: https://github.com/aetperf/edsger/actions/workflows/tests.yml/badge.svg?branch=release
    :alt: Tests Status


======
Edsger
======


    Graph algorithms in Cython


Welcome to our Python library for fast path algorithms on graphs, built with the power of Cython. This library is designed for developers and researchers who need to perform efficient path computations on large graphs, while still having the flexibility and ease of use that Python provides.

The library includes a range of common path algorithms, such as shortest paths. It is also open-source and easy to integrate with other Python libraries, giving you the freedom and flexibility to use it in your existing projects.

To get started, simply install the library using pip, and import it into your Python project. Then, you can start using the library to perform efficient path computations on your graphs.

++++++++++
Algorithms
++++++++++

Dijkstra
--------

.. code:: ipython3

    import pandas as pd
    
    from edsger.path import Dijkstra
    
    edges = pd.DataFrame(
        data={
            "tail": [0, 0, 1, 1, 2],
            "head": [1, 2, 2, 3, 3],
            "weight": [1.0, 2.0, 0.0, 2.0, 1.0],
        }
    )
    
    edges

.. raw:: html

    <div>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>tail</th>
          <th>head</th>
          <th>weight</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0</td>
          <td>1</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0</td>
          <td>2</td>
          <td>2.0</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1</td>
          <td>2</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1</td>
          <td>3</td>
          <td>2.0</td>
        </tr>
        <tr>
          <th>4</th>
          <td>2</td>
          <td>3</td>
          <td>1.0</td>
        </tr>
      </tbody>
    </table>
    </div>


.. code:: python

    sp = Dijkstra(edges, orientation="out")
    path_lengths = sp.run(vertex_idx=0)
    path_lengths


.. parsed-literal::

    array([0., 1., 1., 2.])



