"""
to compare with original results, on 2080ti machine,
you can run

h5diff -vr thesis-yimeng-v1/results/datasets/prepared/crcns_pvc-8_neural.hdf5 thesis-yimeng-v2/results/datasets/prepared/crcns_pvc-8_neural.hdf5

AND

h5diff -vr thesis-yimeng-v1/results/datasets/prepared/crcns_pvc-8_image.hdf5 thesis-yimeng-v2/results/datasets/prepared/crcns_pvc-8_images.hdf5



for 8k, run

$ROOT/scripts/debug/test_prepared_data_yuanyuan8k_another.py
AND

$ROOT/scripts/debug/test_prepared_data_yuanyuan8k.py

since 8k data got changed (check <https://github.com/leelabcnbc/cnn-model-leelab-8000/issues/3>)
"""  # noqa: E501

from thesis_v2.data.prepared import crcns_pvc8, yuanyuan_8k

# this is optional.
crcns_pvc8.natural_data('large', 144, 4, None, read_only=False)
# a = crcns_pvc8.natural_data_legacy('large', 144, 4, None)
# print(a[1].mean())  # this should be different every time as seed=None
crcns_pvc8.natural_data('medium', 144, 4, None, read_only=False)

# 8k data
for group in ('a', 'b', 'c'):
    # this is similar to yuanyuan's own processing.
    # this is kept mostly for debugging purposes.
    yuanyuan_8k.images(group, 256, 128, read_only=False)
    # in practice I should try smaller window size, and more downsampling.

    # 2x downsampling one.
    yuanyuan_8k.images(group, 200, 100, read_only=False)

    # all 4x downsampling.
    # yuanyuan_8k.images(group, 256, 64, read_only=False)
    # only this was used.
    yuanyuan_8k.images(group, 200, 50, read_only=False)

    # yuanyuan_8k.images(group, 160, 40, read_only=False)

    # yuanyuan_8k.images(group, 128, 32, read_only=False)
