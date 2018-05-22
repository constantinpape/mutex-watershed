import unittest
import numpy as np
import mutex_watershed
import vigra
import matplotlib.pyplot as plt


def run_mst(affinities,
            offsets, stride,
            seperating_channel=2,
            invert_dam_channels=True,
            randomize_bounds=True):
    import constrained_mst as cmst
    assert len(affinities) == len(offsets), "%s, %i" % (str(affinities.shape), len(offsets))
    affinities_ = np.require(affinities.copy(), requirements='C')
    if invert_dam_channels:
        affinities_[seperating_channel:] *= -1
        affinities_[seperating_channel:] += 1
    sorted_edges = np.argsort(affinities_.ravel())
    # run the mst watershed
    vol_shape = affinities_.shape[1:]
    mst = cmst.ConstrainedWatershed(np.array(vol_shape),
                                    offsets,
                                    seperating_channel,
                                    stride)
    if randomize_bounds:
        mst.compute_randomized_bounds()
    mst.repulsive_ucc_mst_cut(sorted_edges, 0)
    segmentation = mst.get_flat_label_image().reshape(vol_shape)
    return segmentation


class B2ATest(unittest.TestCase):

    # test mutex watershed clustering on a graph
    # with random edges and random mutex edges
    def test_b2a(self):
        offsets = [[-1, 0], [0, -1], [-15, 0], [0, -15]]

        # all the vigra things to load an image ...
        bmap = vigra.impex.readImage('97010.png').view(np.ndarray).squeeze()
        bmap = 1. - bmap.T

        aff_features = mutex_watershed.boundaries_to_affinities_2d(bmap, offsets)
        self.assertEqual(aff_features.shape[0], 9)
        self.assertEqual(aff_features.shape[1], len(offsets))
        self.assertEqual(aff_features.shape[2:], bmap.shape)

        # get the affinities
        affs = np.zeros((len(offsets),) + bmap.shape)

        # for the nn channels, take the mean (which is the only well defined one!)
        affs[:2] = aff_features[0, :2]

        # for the log range ones we can choose from these features:
        # 0: mean
        # 1: variance
        # 2 - 8: quantiles (including min, max)
        affs[2:] = aff_features[0, 2:]
        affs = 1. - affs

        offsets = np.array(offsets)
        stride = np.array([6, 6])
        segmentation = run_mst(affs, offsets, stride)

        fig, ax = plt.subplots(3)
        ax[0].imshow(bmap, cmap='gray')
        ax[1].imshow(affs[2], cmap='gray')
        ax[2].imshow(segmentation)
        plt.show()


if __name__ == '__main__':
    unittest.main()
