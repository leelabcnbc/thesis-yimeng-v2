from pkgutil import get_data
from shlex import quote
from os.path import join, exists, relpath
from os import chmod, makedirs
from subprocess import run
from itertools import product
from collections import OrderedDict
from typing import Union, Tuple
# this is the only dependency, which will check for Py 3.6 only.
# will not import any fancy package.
from .. import dir_dict

pkg_name = __package__

template = """
{script_header}

# this way slurm's own error can be caught as well.
#SBATCH --output={external_log_out}
#SBATCH --error={external_log_err}

echo {singularity_exec_script} | {singularity_call} |& tee {external_log}
""".strip()


# https://stackoverflow.com/questions/434287/what-is-the-most-pythonic-way-to-iterate-over-a-list-in-chunks  # noqa: E501
def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def gen_inner_loc_location(dirname_relative, log_inner):
    return f"""
from thesis_v2 import dir_dict, join
print(join(dir_dict['trash'],
           {repr(dirname_relative)},
           {repr(log_inner)}), end='')
""".strip()


def submit(script_dict,
           script_header_name,
           singularity_exec_header_name,
           use_slurm=False,
           dirname_relative=None,
           chunk_size=1,
           singularity_call_fname='singularity_call',
           ):
    # create dir
    dirname_abs = join(dir_dict['trash'], dirname_relative)
    assert not exists(dirname_abs)
    makedirs(dirname_abs, exist_ok=False)
    dirname_logs_inner = join(dirname_abs, 'logs_inner')
    makedirs(dirname_logs_inner)
    dirname_logs_inner_rel = relpath(dirname_logs_inner, dir_dict['trash'])
    dirname_logs_outer = join(dirname_abs, 'logs_outer')
    makedirs(dirname_logs_outer)

    script_header = get_data(pkg_name,
                             f'script_header/{script_header_name}').decode()
    singularity_exec_header = get_data(
        pkg_name,
        f'singularity_exec_header/{singularity_exec_header_name}').decode()
    # strip for guaranteed one-line.
    singularity_call = get_data(pkg_name,
                                singularity_call_fname).decode().strip()

    script_full_list = []

    for idx, this_chunk in enumerate(chunker(list(script_dict.items()),
                                             chunk_size)):

        script_this_chunk = []
        # add -s so that .local in $HOME won't interfere with python
        # in singularity.
        for log_inner, script_inner in this_chunk:
            # https://stackoverflow.com/questions/6571435/limit-on-file-name-length-in-bash
            # log_inner cannot be too long.
            # just to play safe. I know there is difference between byte and str.
            # but just some hack.
            assert len(log_inner) < 250, f'length is {len(log_inner)}'

            script_for_this = f"""
OUTPUT_THIS=$(python -s -c {quote(
                gen_inner_loc_location(dirname_logs_inner_rel, log_inner))})
PYTHONUNBUFFERED=1 python -s -c {quote(
                script_inner)} |& tee "${{OUTPUT_THIS}}" &
""".strip()
            script_this_chunk.append(script_for_this)

        script_this_chunk.append('wait')

        script_this_chunk = '\n'.join(script_this_chunk)

        # mandatory echoing hostname, for later debugging.

        singularity_exec_script = f"""
echo "host: $(hostname)"

{singularity_exec_header}

{script_this_chunk}
""".strip()
        script_full = template.format(script_header=script_header,
                                      singularity_exec_script=quote(
                                          singularity_exec_script),
                                      singularity_call=singularity_call,
                                      external_log_out=quote(
                                          join(dirname_logs_outer,
                                               f'{idx}.out')),
                                      external_log_err=quote(
                                          join(dirname_logs_outer,
                                               f'{idx}.err')),
                                      external_log=quote(
                                          join(dirname_logs_outer,
                                               f'{idx}')),
                                      )

        script_full_list.append(script_full)

    filelist = []
    for idx, script_full_this in enumerate(script_full_list):
        fname = join(dirname_abs, f'{idx}.sh')
        filelist.append(fname)
        with open(fname, 'w', encoding='utf-8') as f:
            f.write(script_full_this)
        chmod(fname, 0o755)

    print(f'everything is in {dirname_abs}')
    if use_slurm:
        input('press enter to optionally sbatch all scripts')
        # submit everything.
        for fname_this in filelist:
            run(['sbatch', fname_this], check=True)


def call_script_formatter(
        call_script: str, keys_literal: set,
        **kwargs,
):
    new_dict = dict()
    for k, v in kwargs.items():
        if k in keys_literal:
            assert type(v) is str
            new_dict[k] = v
        else:
            new_dict[k] = repr(v)
    return call_script.format(**new_dict)


class ParamIterator:
    def __init__(self):
        self.data = OrderedDict()

    def add_pair(self, key: Union[str, Tuple[str, ...]], values, late_call: bool = False, replace=False):
        # late call is used to handle expensive calls
        # so values() provides the actual result.
        # also, if a generator is passed in, we should wrap it as a lambda,
        # so that it can be reiterated over and over.
        if not replace:
            assert key not in self.data
        else:
            # assert key in self.data
            try:
                assert key in self.data
            except:
                print(key)
                print(self.data)
                raise AssertionError
        if type(key) is str:
            pass
        elif type(key) is tuple:
            # multi key
            for k in key:
                assert type(k) is str
        else:
            raise NotImplementedError
        self.data[key] = {
            'values': values,
            'late_call': late_call,
        }

    def generate(self, key_predicate=None, ret_predicate=None, extra_keys=None):
        if extra_keys is None:
            extra_keys = dict()
        if key_predicate is None:
            def key_predicate(_):
                return True

        keys_to_check = []
        values_to_check = []

        for k, v in self.data.items():
            if key_predicate(k):
                values_to_check.append(v['values']() if v['late_call'] else v['values'])
                keys_to_check.append(k)

        for vs in product(*values_to_check):
            # this 1-liner cannot handle complicate cases.
            # yield OrderedDict(zip(self.data.keys(), vs))
            # construct the new obj
            assert len(keys_to_check) == len(vs)
            ret_obj = OrderedDict()
            for key_this, value_this in zip(keys_to_check, vs):
                if type(key_this) is str:
                    # convert to singular tuple
                    key_this = (key_this,)
                    value_this = (value_this,)

                assert len(key_this) == len(value_this)
                for kk, vv in zip(key_this, value_this):
                    ret_obj[kk] = vv

            # TODO use `ret_predicate` to filter some ret_obj
            ret_obj.update(extra_keys)
            yield ret_obj
