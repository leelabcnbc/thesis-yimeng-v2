from thesis_v2.configs.model.maskcnn_polished_with_rcnn_k_bl import (
    main_models_8k_validate,
    multipath_models_8k_validate,
    main_models_ns2250_validate,
    multipath_models_ns2250_validate,
    ablation_models_8k_validate,
    ablation_models_ns2250_validate,
    main_models_8k_generator,
    main_models_ns2250_generator,
    multipath_models_8k_generator,
    multipath_models_ns2250_generator,
    ablation_models_8k_generator,
    ablation_models_ns2250_generator,
    ablation_ff_models_8k_validate,
    ablation_ff_models_8k_generator,
    ablation_ff_models_ns2250_generator,
    ablation_ff_models_ns2250_validate,
    ablation_7_models_8k_generator,
    ablation_7_models_8k_validate,
    ablation_7_models_ns2250_generator,
    ablation_7_models_ns2250_validate,
)

from thesis_v2.configs.model.maskcnn_polished_with_rcnn_k_bl import keygen


def check_no_overlap(*generators):
    set_list = []
    for generator in generators:
        set_this = set()
        for x in generator(with_source=False):
            key = keygen(
                # skip these two because they are of float
                **{k: v for k, v in x.items() if k not in {'scale', 'smoothness'}}
            )
            assert key not in set_this
            set_this.add(key)
        set_list.append(set_this)
    assert len(set.union(*set_list)) == sum(len(z) for z in set_list)


if __name__ == '__main__':
    main_models_8k_validate()
    multipath_models_8k_validate()
    main_models_ns2250_validate()
    multipath_models_ns2250_validate()
    ablation_models_8k_validate()
    ablation_models_ns2250_validate()
    ablation_ff_models_8k_validate()
    ablation_ff_models_ns2250_validate()
    ablation_7_models_8k_validate()
    ablation_7_models_ns2250_validate()
    check_no_overlap(
        main_models_8k_generator,
        main_models_ns2250_generator,
        multipath_models_8k_generator,
        multipath_models_ns2250_generator,
        ablation_models_8k_generator,
        ablation_models_ns2250_generator,
        # there is some overlap between
        # ablation_ff_models and
        # ablation_models_8k
        ablation_ff_models_8k_generator,
        ablation_ff_models_ns2250_generator,
        ablation_7_models_8k_generator,
        ablation_7_models_ns2250_generator,
    )
