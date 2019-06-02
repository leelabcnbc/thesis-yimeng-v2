from pkgutil import get_data
from shlex import quote
from os.path import join, exists, relpath
from os import chmod, makedirs
from subprocess import run
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
