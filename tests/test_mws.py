import unittest
import numpy as np
import mutex_watershed


class MwsTest(unittest.TestCase):

    # test mutex watershed clustering on a graph
    # with random edges and random mutex edges
    def test_mws_clustering_random_graph(self):
        number_of_labels = 500
        number_of_edges = 1000
        number_of_mutex_edges = 2000

        # random edges
        edges = np.random.randint(0, number_of_labels,
                                  size=(number_of_edges, 2),
                                  dtype='uint64')
        # filter for redundant entries
        edge_mask = edges[:, 0] != edges[:, 1]
        edges = edges[edge_mask]

        # random mutex edges
        mutex_edges = np.random.randint(0, number_of_labels,
                                        size=(number_of_mutex_edges, 2),
                                        dtype='uint64')
        # filter for redundant entries
        edge_mask = mutex_edges[:, 0] != mutex_edges[:, 1]
        mutex_edges = mutex_edges[edge_mask]

        # random weights
        edge_weights = np.random.rand(edges.shape[0])
        mutex_weights = np.random.rand(mutex_edges.shape[0])

        # compute mutex labeling
        node_labels = mutex_watershed.compute_mws_clustering(number_of_labels,
                                                             edges, mutex_edges,
                                                             edge_weights, mutex_weights)
        self.assertEqual(len(node_labels), number_of_labels)


if __name__ == '__main__':
    unittest.main()
