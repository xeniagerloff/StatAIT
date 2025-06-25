import argparse
import numpy as np
import json
from standard_tests.attribute_test import AttributeTest
from standard_tests.membership_test import MembershipTest


def run_tests(args):
    train_data = np.load(args.data_dir + "/train_data.npy")
    test_data = np.load(args.data_dir + "/test_data.npy")
    synth_data = np.load(args.data_dir + "/synth_data.npy")

    results = {}

    membership_test = MembershipTest(train_data, test_data, synth_data, seed=args.seed)
    membership_results = membership_test.run()
    results.update(membership_results)

    most_freq_attr_test = AttributeTest(
        train_data,
        synth_data,
        test_data,
        num_known=args.num_known,
        type="most_freq",
        seed=args.seed,
    )
    most_freq_attr_results = most_freq_attr_test.run()
    results.update(most_freq_attr_results)

    random_attr_test = AttributeTest(
        train_data,
        synth_data,
        test_data,
        num_known=args.num_known,
        type="random",
        seed=args.seed,
    )
    random_attr_results = random_attr_test.run()
    results.update(random_attr_results)

    return results


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
        "--num_known", default=None, help="No. of known attributes in attribute test"
    )
    parser.add_argument("--seed", default=None, help="Random seed")

    args = parser.parse_args()

    results = run_tests(args)
    with open(args.output_dir + "/standard_results.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
