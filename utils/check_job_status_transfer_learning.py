"""
usage:
check_job_status_transfer_learning.py /root/path/to/models n_neuron

redirect (>) the output to some file, remove the last line, and you will be
good.
"""

from sys import argv, version_info
from os import walk
from os.path import isabs, split, join
from shlex import quote
from collections import defaultdict

assert version_info >= (3, 6)

files_to_check_gpu = {'stats_best.json',
                      'stats_all.json',
                      'config.json',
                      'config_extra.json',
                      'best.pth'}


def main():
    parent_dict = defaultdict(lambda: (set(), set()))

    (root_dir, num_neuron) = argv[1:]
    num_neuron = int(num_neuron)

    ref_set = {f'n{idx}' for idx in range(num_neuron)}

    files_to_check = files_to_check_gpu

    assert isabs(root_dir)
    for idx, (root, dirs, files) in enumerate(walk(root_dir)):
        # if idx % 100 == 0:
        #     print(idx)
        if len(dirs) != 0:
            # it's not root folder.
            continue
        parent, neural_idx_this = split(root)
        if not (files_to_check - set(files)):
            # this is a finished one.
            assert neural_idx_this in ref_set
            # first is the good one.
            parent_dict[parent][0].add(neural_idx_this)
        else:
            # this is a bad one.
            parent_dict[parent][1].add(neural_idx_this)

        del parent
        del neural_idx_this

    good = bad = bad_detailed = 0
    for x, s in parent_dict.items():
        if s[0] != ref_set:
            bad += 1
            for subfolder in sorted(s[1]):
                print(f'rm -r {quote(join(x, subfolder))}')
                bad_detailed += 1
        else:
            good += 1

    print('good', good, 'bad', bad, 'bad_detailed', bad_detailed)


if __name__ == '__main__':
    main()
