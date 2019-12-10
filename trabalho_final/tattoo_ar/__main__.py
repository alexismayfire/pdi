import sys

from tattoo_ar.run import run


def main(args=None):
    """The main routine."""
    if args is None:
        args = sys.argv[1:]

    run()

if __name__ == "__main__":
    main()