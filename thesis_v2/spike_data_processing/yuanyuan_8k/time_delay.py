import numpy as np

from .response_extraction import extract_response


def find_delay(*,
               time_delays_to_try,
               spike_count_list,
               param_id_list,
               mapping_record_paras,
               frame_per_image,
               duration_per_frame,
               normaliztion_config=None,
               delay_metric='legacy',
               extraction_length,
               ):
    # get data one by one.
    assert len(spike_count_list) == len(param_id_list) > 1
    num_neuron, num_condition = spike_count_list[0].shape[:2]

    correlations_all = []

    for time_delay_idx, time_delay_this in enumerate(time_delays_to_try):
        response_all_this_delay = []
        for spike_count, param_id in zip(spike_count_list, param_id_list):
            # get the shuffle idx
            assert spike_count.ndim == 3
            assert spike_count.shape[:2] == (num_neuron, num_condition,)
            sort_idx = mapping_record_paras[param_id, 0][0, 1].ravel()
            assert sort_idx.shape == (frame_per_image * num_condition,)
            # assert np.array_equal(np.sort(sort_idx), np.arange(1, frame_per_image * num_condition + 1))
            sort_idx = np.argsort(sort_idx)

            response_all_this_delay.append(
                extract_response(
                    spike_count=spike_count,
                    sort_index=sort_idx,
                    # convert to int, so that round() method does not get overloaded (producing a float)
                    time_delay=time_delay_this,
                    num_frame=frame_per_image,
                    # althougth yuanyuan's code writes 60, it's effectively 61
                    extraction_length=extraction_length,
                    frame_time=duration_per_frame,
                    normaliztion_config=normaliztion_config,
                )
            )
        response_all_this_delay = np.asarray(response_all_this_delay)
        # num_trial x num_neuron x num_image
        assert response_all_this_delay.shape == (len(spike_count_list), num_neuron, frame_per_image * num_condition)

        # then compute correlation.
        correlations_this = []
        for idx_neuron in range(response_all_this_delay.shape[1]):
            if delay_metric == 'legacy':
                corr_this = np.corrcoef(response_all_this_delay[:, idx_neuron], rowvar=True)
                # take upper triangular.
                # TODO: check number of NaNs.
                #   right now this is very strict.
                assert np.all(np.isfinite(corr_this))
                # print(corr_this.shape)
                index_this = np.triu_indices(len(spike_count_list), 1)
                # print(index_this)
                correlations_this.append(np.nanmean(corr_this[index_this]))
            else:
                raise NotImplementedError
        correlations_all.append(correlations_this)

    correlations_all = np.asarray(correlations_all)
    # best offset index
    best_delay_index = np.argmax(correlations_all, axis=0)
    best_correlation = correlations_all[best_delay_index, np.arange(num_neuron)]
    best_delay = np.asarray(time_delays_to_try)[best_delay_index]

    return {
        'best_correlation': best_correlation,
        'best_delay': best_delay,
    }
