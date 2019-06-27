"""
usage:
check_job_status_maskcnn_like.py /root/path/to/models

redirect (>) the output to some file, remove the last line, and you will be
good.
"""

from sys import argv, version_info
from os import walk
from os.path import isabs
from shlex import quote

assert version_info >= (3, 6)

files_to_check_gpu = {'stats_best.json',
                      'stats_all.json',
                      'config.json',
                      'config_extra.json',
                      'best.pth'}


def main():
    root_dir, = argv[1:]

    files_to_check = files_to_check_gpu

    assert isabs(root_dir)
    bad_dirs = []
    bad = good = 0
    for root, dirs, files in walk(root_dir):
        if len(dirs) != 0:
            # it's not root folder.
            continue
        if files_to_check - set(files):
            bad += 1
            bad_dirs.append(root)
            # there can be false positives (pure empty folders); but that's fine & harmless.
        else:
            good += 1

    for x in bad_dirs:
        print(f'rm -rf {quote(x)}')

    print('good', good, 'bad', bad)


if __name__ == '__main__':
    main()
