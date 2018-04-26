#include "pybind11/pybind11.h"

#include "xtensor/xmath.hpp"
#include "xtensor/xarray.hpp"

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pytensor.hpp"

#include <iostream>
#include <numeric>
#include <cmath>

#include "mutex_watershed/mutex_watershed.hxx"

namespace py = pybind11;

PYBIND11_MODULE(mutex_watershed, m)
{
    xt::import_numpy();

    m.doc() = R"pbdoc(
        The Mutex Watershed for image segmentation

        .. currentmodule:: mutex_watershed

        .. autosummary::
           :toctree: _generate

           example1
           example2
           readme_example1
           vectorize_example1
    )pbdoc";

    m.def("compute_mws_clustering", compute_mws_clustering, "Compute mutex watershed segmentation on a graph");
}
