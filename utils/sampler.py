import random

class Sampler:
    """
    Class for performing different sampling methods on the given dataset
    """

    def __init__(self, method='stratified', k=0, percentage=0.25, seed=2019):
        """
        Initialize the sampler with specified sampling method, percentage, and random seed
        """
        self.method = method  # sampling approach (e.g., 'stratified', 'bootstrap')
        self.percentage = percentage  # sampling percentage
        self.k = k  # number of folds for k-fold sampling
        random.seed(seed)  # set random seed for reproducibility
    
    def sample_by_percentage(self, indices, labels):
        """
        Perform sampling based on the specified percentage of the dataset
        returns a list of sampled indices
        """
        if len(indices) != len(labels):
            raise ValueError("Length of indices and labels must match")

        if self.method == 'stratified':
            sampled_indices = self._stratified_sampling(indices, labels)
        elif self.method == 'bootstrap':
            sampled_indices = self._bootstrap_sampling(indices)
        else:
            raise ValueError(f"Sampling method '{self.method}' is not supported")

        return sampled_indices

    def sample_by_k_fold(self, indices, labels):
        """
        Perform k-fold sampling while preserving class distribution
        returns a list of k subsets, each containing 1/k of the dataset
        """
        if self.k <= 1 or self.k > len(indices):
            raise ValueError("k must be greater than 1 and less than or equal to the dataset size")

        if len(indices) != len(labels):
            raise ValueError("Length of indices and labels must match")

        k_fold_indices = self._k_fold_stratified(indices, labels)
        return k_fold_indices

    def sample_by_k_fold_pure(self, indices):
        """
        Perform pure k-fold cross-validation without preserving class distribution
        returns a list of k subsets, each containing one fold of the dataset
        """
        if self.k <= 1 or self.k > len(indices):
            raise ValueError("k must be greater than 1 and less than or equal to the dataset size")

        k_fold_indices = self._k_fold_random(indices)
        return k_fold_indices    

    def _stratified_sampling(self, indices, labels):
        """
        Perform stratified sampling, maintaining class distribution in the sampled set
        """
        positive_indices = [indices[i] for i in range(len(indices)) if labels[i] != 0]
        negative_indices = [indices[i] for i in range(len(indices)) if labels[i] == 0]

        pos_sample_count = int(len(positive_indices) * self.percentage)
        neg_sample_count = int(len(negative_indices) * self.percentage)

        pos_samples = random.sample(positive_indices, pos_sample_count)
        neg_samples = random.sample(negative_indices, neg_sample_count)

        return pos_samples + neg_samples
    
    def _bootstrap_sampling(self, indices):
        """
        Perform bootstrap sampling with replacement, sampling up to percentage size of the dataset
        returns a list of sampled indices, possibly with duplicates
        """
        sample_size = int(len(indices) * self.percentage)
        return [indices[random.randint(0, len(indices) - 1)] for _ in range(sample_size)]

    def _k_fold_stratified(self, indices, labels):
        """
        Perform stratified k-fold sampling, preserving class distribution in each fold
        returns a list of k subsets, each containing a balanced portion of the dataset
        """
        pos_indices = [i for i in range(len(indices)) if labels[i] != 0]
        neg_indices = [i for i in range(len(indices)) if labels[i] == 0]

        pos_fold_size = len(pos_indices) // self.k
        neg_fold_size = len(neg_indices) // self.k

        pos_remainder = len(pos_indices) % self.k
        neg_remainder = len(neg_indices) % self.k

        k_fold_indices = []
        for _ in range(self.k):
            fold_indices = []
            
            # distribute positive samples across folds
            pos_sample_count = pos_fold_size + (1 if pos_remainder > 0 else 0)
            pos_remainder -= 1 if pos_remainder > 0 else 0
            fold_indices.extend(random.sample(pos_indices, pos_sample_count))
            pos_indices = [idx for idx in pos_indices if idx not in fold_indices]
            
            # distribute negative samples across folds
            neg_sample_count = neg_fold_size + (1 if neg_remainder > 0 else 0)
            neg_remainder -= 1 if neg_remainder > 0 else 0
            fold_indices.extend(random.sample(neg_indices, neg_sample_count))
            neg_indices = [idx for idx in neg_indices if idx not in fold_indices]

            k_fold_indices.append(fold_indices)

        return k_fold_indices

    def _k_fold_random(self, indices):
        """
        Perform random k-fold sampling without preserving class distribution
        returns a list of k subsets of the dataset
        """
        fold_size = len(indices) // self.k
        remainder = len(indices) % self.k

        random_indices = list(indices)  # copy the original list
        random.shuffle(random_indices)

        k_fold_indices = []
        for _ in range(self.k):
            current_fold_size = fold_size + (1 if remainder > 0 else 0)
            remainder -= 1 if remainder > 0 else 0

            fold = random_indices[:current_fold_size]
            random_indices = random_indices[current_fold_size:]
            k_fold_indices.append(fold)

        return k_fold_indices
