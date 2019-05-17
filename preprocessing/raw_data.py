"""
to compare with original results, on 2080ti machine,
you can run things like

h5diff -vr thesis-yimeng-v1/results/datasets/raw/crcns_pvc-8_neural.hdf5 thesis-yimeng-v2/results/datasets/raw/crcns_pvc-8_neural.hdf5

to check (there are 5 files in total, for crcns_pvc8 and yuanyuan_8k)
"""  # noqa: E501

from thesis_v2.data import raw

raw.save_data()
