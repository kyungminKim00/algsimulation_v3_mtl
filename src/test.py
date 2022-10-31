from contextlib import contextmanager
import sys
from util import print_flush


for i in range(100):
    # sys.stdout.write(
    #     "\r>> [%d] Converting data" % (i)
    # )
    # sys.stdout.flush()

    print_flush("\r>> test: {}".format(i))
