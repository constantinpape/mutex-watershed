#pragma once
#include "xtensor/xtensor.hpp"
// TODO import vigra accumulators


namespace mutex_watershed {


    // TODO this could be done more elegantly e.g. using views
    template<class BOUNDARIES, class AFF_TYPE>
    inline void accumulate_affinity(const xt::xexpression<BOUNDARIES> & boundaries_exp,
                                    const int64_t axis,
                                    const int64_t pos_i,
                                    const int64_t pos_j,
                                    const int diff,
                                    const bool dir,
                                    AFF_TYPE & aff) {
        typedef typename BOUNDARIES::value_type BoundaryType;
        typedef AFF_TYPE AffinityType;
        const auto & boundaries = boundaries_exp.derived_cast();
        // TODO replace with vigra accumulator
        std::vector<BoundaryType> values;

        // accumulate along x
        if(axis == 0) {
            // dir encodes whether we iterate into the positive or
            // negative direction
            if(dir) {
                for(int64_t i = pos_i; i <= pos_i + diff; ++i) {
                    values.push_back(boundaries(i, pos_j));
                }
            } else {
                for(int64_t i = pos_i; i >= pos_i - diff; --i) {
                    values.push_back(boundaries(i, pos_j));
                }
            }
        }

        // accumulate along y
        else {
            // dir encodes whether we iterate into the positive or
            // negative direction
            if(dir) {
                for(int64_t j = pos_j; j <= pos_j + diff; ++j) {
                    values.push_back(boundaries(pos_i, j));
                }
            } else {
                for(int64_t j = pos_i; j >= pos_j - diff; --j) {
                    values.push_back(boundaries(pos_i, j));
                }
            }
        }

        // TODO replace with vigra accumulation
        aff = static_cast<AffinityType>(*std::max_element(values.begin(), values.end()));
    }

    // important !
    // for now we only support straight offsets !
    // TODO use vigra accumulator and feature !
    template<class BOUNDARIES, class AFFINITIES>
    void boundaries_to_affinities_2d(const xt::xexpression<BOUNDARIES> & boundaries_exp,
                                     const std::vector<std::array<int, 2>> & offsets,
                                     xt::xexpression<AFFINITIES> & affinities_exp,
                                     const std::string & feature="max") {
        const auto & boundaries = boundaries_exp.derived_cast();
        auto & affinities = affinities_exp.derived_cast();

        typedef typename BOUNDARIES::value_type BoundaryType;
        typedef typename AFFINITIES::value_type AffinityType;

        const auto & shape = boundaries.shape();

        // number of channels is equal to the length of the offsets
        // (and affinities.shape()[0])
        const size_t n_channels = offsets.size();
        for(int64_t c = 0; c < n_channels; ++c) {
            const auto & offset = offsets[c];
            for(int64_t i = 0; i < shape[0]; ++i) {
                for(int64_t j = 0; j < shape[1]; ++j) {
                    // TODO do we clip or skip invalid offsets ?
                    // for now I decided to clip
                    const int64_t i_off = std::min(std::max(i + off[0], 0), shape[0]);
                    const int64_t j_off = std::min(std::max(j + off[1], 0), shape[1]);

                    AffinityType aff;
                    // accumulate along x axis
                    if(i_off != i && j_off == j) {
                        accumulateAffinity(boundaries, 0, i, j, std::abs(i_off - i), i_off > i, aff);
                    }
                    // accumulate along y axis
                    else if(i_off == i && j_off != j) {
                        accumulateAffinity(boundaries, 1, i, j, std::abs(j_off - j), j_off > j, aff);
                    }
                    // don't accumulate if offsets point to the same pixel
                    // (this happens for boundary pixels)
                    else if(i_off == i && j_off == j) {
                        aff = 0;
                    }
                    // we don't support diagonal offsets (yet)
                    else if(i_off != i && j_off != j) {
                        throw std::runtime_error("Diagonal offsets are not supported!");
                    }

                    affinities(c, i, j) = aff;
                }
            }
        }
    }


}
