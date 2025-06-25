import argparse
import numpy as np
from stat_ait.test import PartialKnowledgeTester
from stat_ait.estimate import EstimateCollector
from stat_ait.utils import get_data_batch


def run_test(args):
    train_data = np.load(args.data_dir + "/train_data.npy")
    test_data = np.load(args.data_dir + "/test_data.npy")
    synth_data = np.load(args.data_dir + "/synth_data.npy")

    # Optionally, batch train data
    if hasattr(args, "n_batches") and args.n_batches > 1:
        name = f"batch{args.batch_idx + 1}_of_{args.n_batches}"
    else:
        name = ""
    train_batch, test_data_plus_batch_complement = get_data_batch(
        args.batch_idx, train_data, test_data, args.n_batches
    )

    # Sample and test random partial knowledge options for each patient in the train set
    tester = PartialKnowledgeTester(
        train_batch,
        synth_data,
        test_data=test_data_plus_batch_complement,
        max_known=0.5,
        max_workers=args.max_workers,
        attack_model="filter",
        seed=args.seed,
    )
    tester.run(
        min_filters=args.min_filters,
        path=args.output_dir + f"/tester_{name}",
        verbose=True,
    )
    # Compute bootstrapped upper bounds for expected additional success
    estimate_collector = EstimateCollector(
        tester,
        max_workers=args.max_workers,
        n_bootstrap=5000,
        alpha=0.95,
        chunk_size=int(4e5),
    )
    df_estimates = estimate_collector.run()
    df_estimates_updated = estimate_collector.run_updates(
        df_estimates,
        max_updates=args.max_updates,
        n_add=args.n_add,
        uncertainty_thresh=0.025,
    )
    return name, df_estimates_updated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", default=None, help="Path to the binary data directory"
    )
    parser.add_argument(
        "--output_dir", default=None, help="Path to the output directory"
    )
    parser.add_argeument("--exp_name", default="", help="Name of the experiment")
    parser.add_argument(
        "--max_workers", default=1, help="Max. number of CPUs used in parallelization"
    )
    parser.add_argument("--seed", default=None, help="Random seed")

    # Large training sets may benefit from batched processing
    parser.add_argument(
        "--n_batches", default=1, help="No. of batches for processing the training set"
    )
    parser.add_argument("--batch_idx", default=0, help="Index of batch to be processed")

    # Test hyperparameters may need to be increased if uncertainty of estimation is too high
    parser.add_argument(
        "--min_filters",
        default=50,
        help="Min. number of sampled filters per patient and filter size",
    )
    parser.add_argument(
        "--max_updates",
        default=10,
        help="Max. number of update iterations to decrease uncertainty",
    )
    parser.add_argument(
        "--n_add",
        default=50,
        help="Number of filters added during each update iteration",
    )

    args = parser.parse_args()

    name, df_estimates = run_test(args)
    df_estimates.to_csv(args.output_dir + f"/estimates_{name}.csv", index=False)


if __name__ == "__main__":
    main()
