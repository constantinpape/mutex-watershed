#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "xtensor/xmath.hpp"
#include "xtensor/xarray.hpp"

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pytensor.hpp"

#include <iostream>
#include <numeric>
#include <cmath>

#include "mutex_watershed/mutex_watershed.hxx"
#include "mutex_watershed/boundaries_to_affinities.hxx"

namespace py = pybind11;

namespace mutex_watershed {

    void export_mws_clustering(py::module & m) {
        m.def("compute_mws_clustering",[](const uint64_t number_of_labels,
                                          const xt::pytensor<uint64_t, 2> & uvs,
                                          const xt::pytensor<uint64_t, 2> & mutex_uvs,
                                          const xt::pytensor<float, 1> & weights,
                                          const xt::pytensor<float, 1> & mutex_weights){
            xt::pytensor<uint32_t, 1> node_labeling = xt::zeros<uint64_t>({(int64_t) number_of_labels});
            {
                py::gil_scoped_release allowThreads;
                compute_mws_clustering(number_of_labels, uvs,
                                       mutex_uvs, weights,
                                       mutex_weights, node_labeling);
            }
            return node_labeling;
        }, py::arg("number_of_labels"),
           py::arg("uvs"), py::arg("mutex_uvs"),
           py::arg("weights"), py::arg("mutex_weights"));
    }


    void export_boundaries_to_affinities(py::module & m) {
        m.def("boundaries_to_affinities_2d",[](const xt::pytensor<float, 2> & boundaries,
                                               const std::vector<std::array<int, 2>> & offsets){
            int64_t n_channels = offsets.size();
            int64_t sx = boundaries.shape()[0];
            int64_t sy = boundaries.shape()[1];
            int64_t n_feats = 9;
            xt::pytensor<float, 4> affinities = xt::zeros<float>({n_feats, n_channels, sx, sy});
            {
                py::gil_scoped_release allowThreads;
                boundaries_to_affinities_2d(boundaries, offsets, affinities);
            }
            return affinities;
        }, py::arg("boundaries"), py::arg("offsets"));
    }


}


PYBIND11_MODULE(mws, m)
{
    xt::import_numpy();

    m.doc() = R"pbdoc(
        The Mutex Watershed for image segmentation

        .. currentmodule:: mutex_watershed

        .. autosummary::
           :toctree: _generate

           compute_mws_clustering
    )pbdoc";

    mutex_watershed::export_mws_clustering(m);
    mutex_watershed::export_boundaries_to_affinities(m);
}
