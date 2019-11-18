

def get_config(*,
               device,
               max_epoch,
               val_test_every,
               early_stopping_field,
               show_every
               ):
    phase1_dict = {
        'max_epoch': max_epoch,
        'lr_config': None,
        'early_stopping_config': {'patience': 10} if (early_stopping_field is not None) else None,
    }

    phase2_dict = {
        'max_epoch': max_epoch,
        'lr_config': {'type': 'reduce_by_factor', 'factor': 1 / 3},
        'early_stopping_config': {'patience': 10} if (early_stopping_field is not None) else None,
    }

    phase3_dict = {
        'max_epoch': max_epoch,
        'lr_config': {'type': 'reduce_by_factor', 'factor': 1 / 3},
        'early_stopping_config': {'patience': 10} if (early_stopping_field is not None) else None,
    }

    phase_config_dict_list = [phase1_dict, phase2_dict, phase3_dict]
    global_config_dict = {
        'device': device,
        'loss_every_iter': 1,
        'val_every': val_test_every,  # 256 is about 12800 stimuli.
        'test_every': val_test_every,
        # 'output_loss' is what we care actually,
        # such as MSE, or corr, etc.
        'early_stopping_field': early_stopping_field,
        'show_every': show_every,
    }

    return {
        'global': global_config_dict,
        'per_phase': phase_config_dict_list,
    }
