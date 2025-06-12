import sys
from righttyper.righttyper import cli

if __name__ == "__main__":
    # backwards compatibility for subcommand-less '--type-coverage (by-directory|by-file|summary) path'
    if '--type-coverage' in sys.argv:
        i = sys.argv.index('--type-coverage')
        sys.argv[i:i+1] = ['coverage', '--type']

    else:
        # backwards compatibility for subcommand-less run & process
        first_nonopt = next(iter((arg for arg in sys.argv[1:] if not arg.startswith("-"))), None)
        if first_nonopt not in cli.commands and '--help' not in sys.argv:
            sys.argv[1:1] = ['run', '--process']

    cli()
