import random
from utils.sampler import Sampler

class Dataset:
    """
    class for loading, preprocessing, and managing dataset splits
    """

    def __init__(self, file_path=None, file_pointer=None, map_file=None, separator=' ', train_ratio=0.75, test_ratio=0, seed=2019, sampling_method='stratified'):
        """
        initialize the dataset class, load data, and optionally split into train and test sets
        """
        self.feature_names = None  # list of feature names from the dataset
        self.feature_index_map = None  # maps feature names to column indices
        self.train_samples = None  # main training samples
        self.test_samples = None  # test samples, if test_ratio > 0
        self.feature_values = None  # unique values for each feature
        self.original_value_map = {}  # optional map of original values for features
        self.file_path = file_path
        self.map_file = map_file
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.sampling_method = sampling_method

        assert 0 <= self.test_ratio < 1, "test ratio must be between 0 and 1"
        assert self.train_ratio + self.test_ratio <= 1, "train and test ratios combined must be <= 1"

        # load data if file path or file pointer is provided
        if file_path:
            with open(file_path, 'r') as fp:
                self._parse_data(fp, separator)
        elif file_pointer:
            self._parse_data(file_pointer, separator)

        # load original values if a mapping file is provided
        if self.map_file:
            self._load_original_values()

    def _parse_data(self, file_pointer, separator):
        """
        parse the input dataset file and split into training and test samples
        """
        lines = file_pointer.readlines()

        print(lines)
        
        # parse header for feature names
        self.feature_names = lines[0].strip().split(separator)
        self.feature_values = [set() for _ in self.feature_names]
        del lines[0]  # remove header row

        # extract labels from each sample and parse into integer values
        labels = [int(line.strip().split(separator)[-1]) for line in lines]
        sampler = Sampler(method=self.sampling_method, percentage=self.train_ratio, seed=self.seed)
        selected_indices = sampler.sample_by_percentage(range(len(lines)), labels)

        # separate samples for training and testing
        self.train_samples = [self._process_sample(lines[i], separator) for i in selected_indices]
        self.test_samples = [self._process_sample(lines[i], separator) for i in range(len(lines)) if i not in selected_indices]

        # create feature index mapping
        self.feature_index_map = {name: i for i, name in enumerate(self.feature_names)}

    def _process_sample(self, line, separator):
        """
        convert a line from the dataset into a list of integer feature values
        """
        return [int(value) for value in line.strip().split(separator)]

    def _load_original_values(self):
        """
        load original feature values from a separate mapping file
        """
        with open(self.map_file, 'r') as map_fp:
            for line in map_fp:
                feature_value, binary_encoding = line.strip().split(',')
                feature, value = feature_value.split(':')

                for i, bit in enumerate(binary_encoding):
                    binary_feature = f"{feature}:b{i + 1}"
                    binary_value = self.feature_value_map.direct[(binary_feature, '1')]

                    if binary_value not in self.original_value_map:
                        self.original_value_map[binary_value] = [feature]

                    if -binary_value not in self.original_value_map:
                        self.original_value_map[-binary_value] = [feature]

                    self.original_value_map[binary_value if bit == '1' else -binary_value].append(value)

    def get_positive_train_samples(self):
        """
        extract positive samples from a list of samples
        """
        return [(i + 1, sample) for i, sample in enumerate(self.train_samples) if sample[-1] == 1]
    
    def get_negative_train_samples(self):
        """
        extract negative samples from a list of samples
        """
        return [(i + 1, sample) for i, sample in enumerate(self.train_samples) if sample[-1] == 0]
