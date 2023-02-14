import argparse
from pathlib import Path
from utils.dotdict import DotDict


def parse_cmdline_args(parser, path_type=Path):
    assert parser is not None
    parser.add_argument("--config", type=path_type, default="configs/config.yaml")
    parser.add_argument("--model-config", type=path_type, default="configs/wolp_params.yaml")
    parser.add_argument("--benchmark-config", type=path_type, default="configs/benchmark/tpcc.yaml")
    parser.add_argument("--data-snapshot-path", type=path_type, default=None)
    parser.add_argument("--benchbase-config-path", type=path_type, default=None)
    parser.add_argument("--agent", choices=["wolp"], default="wolp", help="Which agent to utilize.")
    parser.add_argument("--seed", type=int, default=-1, help="Seed.")

    # Config files to load from the HPO perspective.
    parser.add_argument("--hpo-config", type=path_type, default="configs/config.yaml")
    parser.add_argument("--hpo-benchmark-config", type=path_type, default=None)

    parser.add_argument("--max-iterations", type=int, default=100, help="Maximum number of iterations.")
    parser.add_argument("--duration", type=float, default=5, help="Maximum duration in hours.")

    parser.add_argument("--workload-timeout", type=int, default=0, help="Maximum workload evaluation time.")
    parser.add_argument("--oltp-num-terminals", type=int, default=0, help="Number of terminals for OLTP.")
    parser.add_argument("--oltp-duration", type=int, default=300, help="Duration to run OLTP workload sample for.")
    parser.add_argument("--oltp-sf", type=int, default=10, help="Default Scale Factor")
    parser.add_argument("--oltp-warmup", type=int, default=0, help="Warmup for OLTP")

    parser.add_argument("--horizon", type=int, default=1, help="Number of steps before resetting (number continuous actions to try)")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout that should be applied on a per-query basis.")
    parser.add_argument("--target", choices=["tps", "latency"], default="tps", help="Which target to optimize for.")
    parser.add_argument("--reward", choices=["multiplier", "relative", "cdb_delta"], default="multiplier", help="Which reward metric to use.")
    parser.add_argument("--dump", type=int, default=0)
    parser.add_argument("--load", type=int, default=0)
    args = parser.parse_args()
    return DotDict(args.__dict__)
