from fcdd.runners.bases import ClassesRunner
from fcdd.runners.argparse_configs import DefaultKeenConfig


class KeenConfig(DefaultKeenConfig):
    def __call__(self, parser):
        parser = super().__call__(parser)
        parser.add_argument('--it', type=int, default=1, help='Number of runs per class with different random seeds.')
        return parser


if __name__ == '__main__':
    runner = ClassesRunner(KeenConfig())
    runner.args.logdir += '_keen_'
    runner.run()
    print()

