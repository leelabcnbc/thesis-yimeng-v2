from thesis_v2.configs.model.maskcnn_polished_with_rcnn_k_bl import (
    main_models_8k_validate,
    multipath_models_8k_validate,
    main_models_ns2250_validate,
    multipath_models_ns2250_validate,
    ablation_models_8k_validate,
    ablation_models_ns2250_validate,
)

if __name__ == '__main__':
    main_models_8k_validate()
    multipath_models_8k_validate()
    main_models_ns2250_validate()
    multipath_models_ns2250_validate()
    ablation_models_8k_validate()
    ablation_models_ns2250_validate()
