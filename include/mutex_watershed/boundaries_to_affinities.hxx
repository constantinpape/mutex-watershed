#pragma once
#include "xtensor/xtensor.hpp"
#include "vigra/accumulator.hxx"


namespace mutex_watershed {


    // TODO this could be done more elegantly e.g. using views
    template<class BOUNDARIES, class FEATURE_TYPE>
    inline void accumulate_affinity(const xt::xexpression<BOUNDARIES> & boundaries_exp,
                                    const int64_t axis,
                                    const int64_t pos_i,
                                    const int64_t pos_j,
                                    const int diff,
                                    const bool dir,
                                    FEATURE_TYPE & feat) {
        // typedef typename BOUNDARIES::value_type BoundaryType;
        typedef FEATURE_TYPE FeatureType;

        // build the vigra accumumlator
        namespace acc = vigra::acc;
        typedef acc::UserRangeHistogram<40> SomeHistogram;   //binCount set at compile time
        typedef acc::StandardQuantiles<SomeHistogram > Quantiles;
        typedef typename vigra::MultiArrayShape<2>::type VigraCoord;

        typedef acc::Select<acc::Mean,        // 1
                            acc::Variance,    // 1
                            Quantiles> SelectType; // 7
        typedef acc::StandAloneAccumulatorChain<2, double, SelectType> AccType;
        AccType accumulator;
        
        // set correct histogram
        vigra::HistogramOptions histogram_opt;
        histogram_opt = histogram_opt.setMinMax(0, 1);
        accumulator.setHistogramOptions(histogram_opt);

        // build vigra coordinate
        VigraCoord vc;

        const auto & boundaries = boundaries_exp.derived_cast();
        // TODO replace with vigra accumulator
        // std::vector<BoundaryType> values;

        // accumulate along x
        if(axis == 0) {
            // dir encodes whether we iterate into the positive or
            // negative direction
            if(dir) {
                for(int64_t i = pos_i; i <= pos_i + diff; ++i) {
                    // values.push_back(boundaries(i, pos_j));
                    vc[0] = i;
                    vc[1] = pos_j;
                    accumulator.updatePassN(boundaries(i, pos_j), vc, 1);
                }
            } else {
                for(int64_t i = pos_i; i >= pos_i - diff; --i) {
                    // values.push_back(boundaries(i, pos_j));
                    vc[0] = i;
                    vc[1] = pos_j;
                    accumulator.updatePassN(boundaries(i, pos_j), vc, 1);
                }
            }
        }

        // accumulate along y
        else {
            // dir encodes whether we iterate into the positive or
            // negative direction
            if(dir) {
                for(int64_t j = pos_j; j <= pos_j + diff; ++j) {
                    //values.push_back(boundaries(pos_i, j));
                    vc[0] = pos_i;
                    vc[1] = j;
                    accumulator.updatePassN(boundaries(pos_i, j), vc, 1);
                }
            } else {
                for(int64_t j = pos_j; j >= pos_j - diff; --j) {
                    vc[0] = pos_i;
                    vc[1] = j;
                    accumulator.updatePassN(boundaries(pos_i, j), vc, 1);
                    // values.push_back(boundaries(pos_i, j));
                }
            }
        }

        // TODO replace with vigra accumulation
        // aff = static_cast<AffinityType>(*std::max_element(values.begin(), values.end()));
        feat[0] = get<acc::Mean>(accumulator);
        feat[1] = get<acc::Variance>(accumulator);
        const auto quantiles = get<Quantiles>(accumulator);
        for(auto qi = 0; qi < 7; ++qi) {
            feat[2 + qi] = quantiles[qi];
        }
    }

    // important !
    // for now we only support straight offsets !
    // TODO use vigra accumulator and feature !
    template<class BOUNDARIES, class AFFINITIES>
    void boundaries_to_affinities_2d(const xt::xexpression<BOUNDARIES> & boundaries_exp,
                                     const std::vector<std::array<int, 2>> & offsets,
                                     xt::xexpression<AFFINITIES> & affinities_exp) {
        const auto & boundaries = boundaries_exp.derived_cast();
        auto & affinities = affinities_exp.derived_cast();

        // typedef typename BOUNDARIES::value_type BoundaryType;
        typedef typename AFFINITIES::value_type AffinityType;

        const auto & shape = boundaries.shape();

        // number of channels is equal to the length of the offsets
        // (and affinities.shape()[0])
        const int64_t n_channels = offsets.size();
        for(int64_t c = 0; c < n_channels; ++c) {
            const auto & offset = offsets[c];
            for(int64_t i = 0; i < shape[0]; ++i) {
                for(int64_t j = 0; j < shape[1]; ++j) {
                    // TODO do we clip or skip invalid offsets ?
                    // for now I decided to skip
                    const int64_t i_off = i + offset[0];
                    const int64_t j_off = j + offset[1];

                    // bounds check
                    const bool bounds_i = (i_off < 0) || (i_off > shape[0]);
                    const bool bounds_j = (j_off < 0) || (j_off > shape[1]);
                    if(bounds_i || bounds_j) {
                        affinities(c, i, j) = 0;
                        continue;
                    }

                    std::vector<AffinityType> aff_feat(9);
                    // accumulate along x axis
                    if(i_off != i && j_off == j) {
                        accumulate_affinity(boundaries, 0, i, j, std::abs(i_off - i), i_off > i, aff_feat);
                    }
                    // accumulate along y axis
                    else if(i_off == i && j_off != j) {
                        accumulate_affinity(boundaries, 1, i, j, std::abs(j_off - j), j_off > j, aff_feat);
                    }
                    // we don't support diagonal offsets (yet)
                    // or zero offsets
                    else{
                        throw std::runtime_error("Diagonal or zeros offsets are not supported!");
                    }

                    for(int f = 0; f < 9; ++f) {
                        affinities(f, c, i, j) = aff_feat[f];
                    }
                }
            }
        }
    }


}
