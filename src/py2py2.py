import os
import shutil
import platform
import argparse
from setup import t_dirs


def cp_py2py2(t_dirs, b_pytopy2):
    for dir in t_dirs:
        for fn in os.listdir(dir):
            t_filename = os.path.join(dir, fn)
            if platform.system() == 'Windows':
                t_filename = t_filename.replace('\\', '/')

            if '.pyx' in fn:
                src_filename = t_filename.replace('.pyx', '.py')
                target_name = t_filename.replace('.pyx', '.py2')
                try:
                    if b_pytopy2:
                        a, b = src_filename, target_name
                        os.rename(src_filename, target_name)
                        print('rename {} to {}'.format(src_filename, target_name))
                    else:
                        b, a = src_filename, target_name
                        os.rename(target_name, src_filename)
                        print('rename {} to {}'.format(target_name, src_filename))
                except FileNotFoundError:
                    print('{} is not exist. skip rename the file'.format(a))


def cp_py2pyx(t_dirs, filename=None):
    if filename is None:
        for dir in t_dirs:
            for fn in os.listdir(dir):
                src_filename = os.path.join(dir, fn)
                if platform.system() == 'Windows':
                    src_filename = src_filename.replace('\\', '/')

                if '.py' == fn[-3:]:
                    target_name = src_filename.replace('.py', '.pyx')
                    if os.path.isfile(target_name):
                        shutil.copy(src_filename, target_name)
                        print('copy {} to {}'.format(src_filename, target_name))
                elif '.py2' == fn[-4:]:
                    target_name = src_filename.replace('.py2', '.pyx')
                    if os.path.isfile(target_name):
                        shutil.copy(src_filename, target_name)
                        print('copy {} to {}'.format(src_filename, target_name))
                else:
                    pass
    else:
        target_name = filename.replace('.py', '.pyx')
        if os.path.isfile(target_name):
            if platform.system() == 'Windows':
                filename = filename.replace('\\', '/')
                target_name = target_name.replace('\\', '/')
            shutil.copy(filename, target_name)
            print('copy {} to {}'.format(filename, target_name))
        else:
            assert False, '{} is not a cython module. Declare pyx first {}'.format(filename, target_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    # init
    parser.add_argument('--b_py2py', type=int, default=0)
    parser.add_argument('--b_pytopy2', type=int, default=0)
    parser.add_argument('--b_py2pyx', type=int, default=0)
    parser.add_argument('--filename', type=str, default=None)
    args = parser.parse_args()

    b_py2py = bool(args.b_py2py)
    b_pytopy2 = bool(args.b_pytopy2)
    b_py2pyx = bool(args.b_py2pyx)

    if args.filename is not None:
        filename = './' + args.filename
    else:
        filename = None

    if b_py2py:
        cp_py2py2(t_dirs, b_pytopy2)

    if b_py2pyx:
        cp_py2pyx(t_dirs, filename)