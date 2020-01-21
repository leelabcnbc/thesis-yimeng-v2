from os.path import abspath, join, dirname, realpath

from sys import version_info

assert version_info >= (3, 6), "must be python 3.6 or higher!"

# print("result root is at {}".format(result_root))

# this way, I don't introduce any additional names.

# abspath implies normpath.

# use realpath to solve aliasing on cnbc cluster.
dir_dict = {
    'root': realpath(join(dirname(__file__), '..')),
}

dir_dict['results'] = abspath(join(dir_dict['root'],
                                   'results'))

dir_dict['datasets'] = abspath(join(dir_dict['results'],
                                    'datasets'))
dir_dict['features'] = abspath(join(dir_dict['results'],
                                    'features'))
dir_dict['models'] = abspath(join(dir_dict['results'],
                                  'models'))
dir_dict['analyses'] = abspath(join(dir_dict['results'],
                                    'analyses'))
dir_dict['plots'] = abspath(join(dir_dict['results'],
                                 'plots'))

dir_dict['visualization'] = abspath(join(dir_dict['results'],
                                         'visualization'))

dir_dict['private_data'] = abspath(join(dir_dict['root'],
                                        'private_data'))

dir_dict['private_data_supp'] = abspath(join(dir_dict['root'],
                                             'private_data_supp'))

dir_dict['debug_data'] = abspath(join(dir_dict['root'],
                                      'debug_data'))

dir_dict['trash'] = abspath(join(dir_dict['root'], 'trash'))
