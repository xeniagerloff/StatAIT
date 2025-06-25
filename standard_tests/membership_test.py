import numpy as np
from tqdm import tqdm
from sklearn import metrics


class MembershipTest:
    def __init__(self, train_data, test_data, synth_data, seed=None):
        self.rng = np.random.default_rng(seed)
        self.synth_data = synth_data
        self.attack_dataset_neg = test_data
        self.n_test = len(test_data)
        self.attack_dataset_pos = self._get_subset(train_data, self.n_test)
        self.chunk_size = int(1024**3 / np.prod(synth_data.shape))  # max 1 GB
        min_mem = self.chunk_size * np.prod(synth_data.shape)
        print(
            f"This chunk size ({self.chunk_size}) needs at least {int(min_mem / 1024**2)} MB memory"
        )

    def _get_subset(self, data, size):
        subset_idx = self.rng.choice(len(data), size, replace=False)
        return data[subset_idx]

    def find_min_hamming_rows(self, A, B):
        p = len(A)
        min_distances_all = np.empty(p, dtype=int)

        for start in tqdm(np.arange(0, p, self.chunk_size), leave=False):
            end = min(start + self.chunk_size, p)
            A_chunk = A[start:end]  # shape (chunk_size, n)
            A_expanded = A_chunk[:, np.newaxis, :]  # shape (chunk_size, 1, n)
            B_expanded = B[np.newaxis, :, :]  # shape (1, q, n)

            # Broadcasted shape: (chunk_size, q, n)
            hamming_distances = np.sum(
                A_expanded != B_expanded, axis=2
            )  # shape (chunk_size, q)
            min_distances = np.min(hamming_distances, axis=1)  # shape (chunk_size,)

            min_distances_all[start:end] = min_distances

        return min_distances_all

    def run(self, path="", save=True):
        ds_pos = self.find_min_hamming_rows(self.attack_dataset_pos, self.synth_data)
        ds_neg = self.find_min_hamming_rows(self.attack_dataset_neg, self.synth_data)

        ds = np.concatenate([ds_pos, ds_neg])
        y_true = np.concatenate([np.ones(self.n_test), np.zeros(self.n_test)])

        argsort = np.argsort(ds)
        sorted_y_true = y_true[argsort]
        sorted_y_pred = np.concatenate([np.ones(self.n_test), np.zeros(self.n_test)])
        acc = metrics.accuracy_score(sorted_y_true, sorted_y_pred)
        results = {"acc": acc, "dists": ds, "y_true": y_true}
        print("Membership acc.", np.round(acc, 3))
        if save:
            np.savez_compressed(
                path,
                acc=results["acc"],
                dists=results["dists"],
                y_true=results["y_true"],
            )
        else:
            return results
