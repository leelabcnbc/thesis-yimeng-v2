"""this file holds some magic numbers for 8k data"""

from ... import dir_dict, join

result_root_dir = join(dir_dict['datasets'], 'spike_data_processing', 'yuanyuan_8k')

para_file_mapping_dict = {
    '042318': (1, 2, 3, 4, 5, 6),
    '043018': (1, 2, 3, 4, 5, 6),
    '050718': (1, 2, 3, 5, 6),
    '050918': (1, 2, 3, 4, 5, 6, 7),
    '051018': (1, 2, 3, 4, 5, 6),
    '051118': (1, 2, 3, 4, 5, 6, 7),
}

imageset_mapping_dict = {
    '042318': '8000a',
    '043018': '8000a',
    '050718': '8000b',
    '050918': '8000c',
    '051018': '8000a',
    '051118': '8000b',
}

record_paras_file_mapping_dict = {
    '8000a': 'Record_paras_Mar072018_RP.mat',
    '8000b': 'Record_paras_May072018_RP8001To16000.mat',
    '8000c': 'Record_paras_May072018_RP16001To24000.mat',
}

# number of frames per image
frame_per_image = 16
# this will be used in splitting movies into frames.
duration_per_frame = 1 / 85 * 4 * 1000
# 121-201 (included) in MATLAB, 120 to 201 (excluded) here.
time_delays_to_try = range(120, 201)

extration_length_for_finding_time_delay = 48
extration_length_for_response_computation = 61


def get_file_names(*, flat: bool):
    data_dict = {
        '042318': range(1, 7),
        '043018': range(2, 8),
        '050718': (2, 3, 4, 6, 7,),
        '050918': range(2, 9),
        '051018': range(2, 8),
        '051118': range(2, 9),
    }

    result = dict()

    for prefix, range_list in data_dict.items():
        data_this = [f'{prefix}_{x}.mat' for x in range_list]
        result[prefix] = data_this

    if flat:
        result = sum(result.values(), [])
    return result


def good_channel_unit(
        *,
        filename, is_prefix
):
    """get_good_channel_unit.m"""
    if is_prefix:
        assert filename in get_file_names(flat=False).keys()
    else:
        assert filename in get_file_names(flat=True)
        # get prefix
        filename = filename[:6]

    good_unit_data_for_this_file = {
        '042318': {
            0: (2, 43, 49, 91),
            1: (2, 3, 6, 8, 12, 16, 17, 21, 31, 34, 36, 40, 43, 44, 46, 49, 52, 53, 57, 60, 79, 83, 87, 88, 91),
            # 2: (),
        },
        '043018': {
            0: (91,),
            1: (2, 3, 6, 8, 12, 16, 17, 31, 33, 34, 36, 40, 43, 44, 45, 46, 49, 55, 57, 60, 83, 87, 88, 91),
            2: (36, 37),
        },
        '050718': {
            0: (91,),
            1: (3, 6, 8, 12, 16, 17, 31, 34, 36, 40, 44, 46, 49, 55, 60, 83, 87, 88, 91),
            # 2: (),
        },
        '050918': {
            0: (91,),
            1: (3, 6, 8, 12, 16, 17, 21, 31, 33, 36, 40, 44, 46, 48, 49, 55, 57, 60, 79, 83, 87, 88, 91),
            # 2: (),
        },
        '051018': {
            0: (91,),
            1: (3, 6, 8, 12, 16, 17, 31, 33, 34, 36, 40, 46, 48, 49, 55, 56, 57, 60, 83, 87, 88, 91),
            # 2: (),
        },
        '051118': {
            0: (91,),
            1: (3, 6, 8, 12, 16, 17, 21, 31, 33, 34, 36, 40, 44, 46, 48, 49, 55, 60, 83, 87, 88, 91),
            # 2: (),
        }
    }[filename]

    # return pairs of electrode number and unit number

    result = []

    # this should be true starting from Python 3.6
    assert list(good_unit_data_for_this_file.keys()) == sorted(good_unit_data_for_this_file.keys())
    for unit, electrodes in good_unit_data_for_this_file.items():
        result.extend([(e, unit) for e in electrodes])

    return result
