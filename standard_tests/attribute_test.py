import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm


def analyse_standard_test_results(
    standard_test_results, diff_thresh, return_failed_ids=False, verbose=True
):
    succ_diff = np.nan_to_num(
        standard_test_results["f1_scores_synth"]
        - standard_test_results["f1_scores_test"]
    )
    failed_mask = succ_diff >= diff_thresh
    n_failed = failed_mask.sum()
    n_records = len(succ_diff)
    perc_passed = 1 - n_failed / n_records
    if verbose:
        print(
            "No. of records that failed: ",
            n_failed,
            f", so {np.round(perc_passed, 4) * 100} % of tested records passed",
        )
    if return_failed_ids:
        failed_records = np.flatnonzero(failed_mask)
        return failed_records, perc_passed
    else:
        return perc_passed


class AttributeTest:
    def __init__(
        self,
        train_data,
        synth_data,
        test_data,
        num_known=500,
        type="most_freq",
        known_attr_set=None,
        seed=None,
    ):
        self.n_train = len(train_data)
        self.attr_count = train_data.shape[1]
        self.synth_data = synth_data
        self.real_data = np.concatenate((train_data, test_data))
        self.train_data = train_data
        self.rng = np.random.default_rng(seed)
        if known_attr_set is None:
            self.num_known = self._compute_num_known(num_known)
            self.type = type
            if self.type == "most_freq":
                self.known_attr_set = self._compute_known_attr_most_freq()
            else:
                self.known_attr_set = None
        else:
            self.known_attr_set = known_attr_set
            self.type = "custom"

        self.secret_attr_set = self._compute_secret(self.known_attr_set)
        self.num_secret = self.attr_count - self.num_known

    def _compute_secret(self, known_attr_set):
        if known_attr_set is None:
            return None
        secret_attr_set = np.array(
            [attr for attr in range(self.attr_count) if attr not in known_attr_set],
            dtype=np.uint32,
        )
        return secret_attr_set

    def _compute_num_known(self, num_known):
        if num_known < 1:
            return np.floor(self.attr_count * num_known).astype(np.int64)
        else:
            return num_known

    def _compute_known_attr_most_freq(self):
        count = np.sum(self.train_data, axis=0)
        np.argsort(count)[::-1][: self.num_known]
        return np.argsort(count)[::-1][: self.num_known]

    def _select_match(self, choices, mask_id):
        if mask_id is not None:
            choices_mask_id = np.nonzero(mask_id == choices)[0]
            if len(choices_mask_id) > 0:
                mask = np.ones(len(choices), dtype=bool)
                mask[choices_mask_id] = False
                choices = choices[mask]
        match_idx = self.rng.choice(choices)
        return match_idx

    def _find_closest_overlap(self, train_known, data_known, mask_id):
        dists = np.sum(data_known[:, train_known == 1], axis=-1)
        if mask_id is not None:
            mask = np.ones(len(dists), dtype=bool)
            mask[mask_id] = False
            max_val = dists[mask].max()
        else:
            max_val = dists.max()
        choices = np.flatnonzero(dists == max_val)
        match_idx = self._select_match(choices, mask_id)
        return match_idx

    def _find_closest_hamming(self, train_known, data_known, mask_id):
        dists = np.sum(train_known[np.newaxis, :] != data_known, axis=1)
        if mask_id is not None:
            mask = np.ones(len(dists), dtype=bool)
            mask[mask_id] = False
            min_val = dists[mask].min()
        else:
            min_val = dists.min()
        choices = np.flatnonzero(dists == min_val)
        match_idx = self._select_match(choices, mask_id)
        return match_idx

    def _find_closest(self, train_known, data_known, mask_id):
        if self.type == "random":
            match_idx = self._find_closest_hamming(
                train_known, data_known, mask_id=mask_id
            )
        else:
            match_idx = self._find_closest_overlap(
                train_known, data_known, mask_id=mask_id
            )
        return match_idx

    def _find_closest_synth(self, train_idx, known_attr_set):
        train_known = self.train_data[train_idx, known_attr_set]  # shape: (1, K)
        synth_known = self.synth_data[:, known_attr_set]  # shape: (N_test, K)

        match_idx = self._find_closest(train_known, synth_known, mask_id=None)
        return match_idx

    def _find_closest_test(self, train_idx, known_attr_set):
        train_known = self.train_data[train_idx, known_attr_set]
        real_known = self.real_data[:, known_attr_set]

        match_idx = self._find_closest(train_known, real_known, mask_id=train_idx)
        return match_idx

    def run(self, path="results.npz", save=True):
        f1_scores_test = np.zeros(self.n_train, dtype=np.float32)
        f1_scores_synth = np.zeros(self.n_train, dtype=np.float32)
        y_true = np.zeros(self.n_train * self.num_secret, dtype=np.bool_)
        y_pred_synth = np.zeros(self.n_train * self.num_secret, dtype=np.bool_)
        y_pred_test = np.zeros(self.n_train * self.num_secret, dtype=np.bool_)

        print("Computing metrics...")
        for i in tqdm(
            range(self.n_train),
            desc="Computing metrics",
            unit="records",
            leave=False,
        ):
            if self.type == "random":
                known_attr_set = self.rng.choice(
                    self.attr_count, self.num_known, replace=False
                )
                secret_attr_set = self._compute_secret(known_attr_set)
            else:
                known_attr_set = self.known_attr_set
                secret_attr_set = self.secret_attr_set

            curr_y_true = self.real_data[i, secret_attr_set]
            if not np.any(curr_y_true):
                continue
            synth_match_idx = self._find_closest_synth(
                i,
                known_attr_set,
            )
            test_match_idx = self._find_closest_test(
                i,
                known_attr_set,
            )
            curr_y_pred_synth = self.synth_data[synth_match_idx, secret_attr_set]
            curr_y_pred_test = self.real_data[test_match_idx, secret_attr_set]

            f1_scores_test[i] = f1_score(curr_y_true, curr_y_pred_test)
            f1_scores_synth[i] = f1_score(curr_y_true, curr_y_pred_synth)
            start, end = int(i * self.num_secret), int((i + 1) * self.num_secret)
            y_true[start:end] = curr_y_true
            y_pred_synth[start:end] = curr_y_pred_synth
            y_pred_test[start:end] = curr_y_pred_test
        overall_f1_synth = f1_score(y_true, y_pred_synth)
        overall_f1_test = f1_score(y_true, y_pred_test)
        mean_f1_synth = np.mean(np.nan_to_num(f1_scores_synth))
        mean_f1_test = np.mean(np.nan_to_num(f1_scores_test))
        if self.type == "most_freq":
            print("Mean F1 diff.", np.round(mean_f1_synth - mean_f1_test, 3))
        else:
            print("Overall F1 diff.", np.round(overall_f1_synth - overall_f1_test, 3))
        results = {
            "f1_scores_test": f1_scores_test,
            "f1_scores_synth": f1_scores_synth,
            "y_true": y_true,
            "y_pred_synth": y_pred_synth,
            "y_pred_test": y_pred_test,
            "overall_f1_synth": overall_f1_synth,
            "overall_f1_test": overall_f1_test,
            "mean_f1_synth": mean_f1_synth,
            "mean_f1_test": mean_f1_test,
        }
        if save:
            np.savez_compressed(
                path,
                f1_scores_test=results["f1_scores_test"],
                f1_scores_synth=results["f1_scores_synth"],
                y_true=results["y_true"],
                y_pred_synth=results["y_pred_synth"],
                y_pred_test=results["y_pred_test"],
                overall_f1_synth=results["overall_f1_synth"],
                overall_f1_test=results["overall_f1_test"],
            )
        else:
            return results
