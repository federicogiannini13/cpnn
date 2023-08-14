import scipy
import scipy.stats
from river import synth
import pandas as pd
import numpy as np
import statistics as st



class RandomWalkGenerator:
    def __init__(
        self,
        first_examples_df: pd.DataFrame = None,
        generator: str = "sine",
        generator_obj: synth.RandomTree = None,
        features_range: tuple = (0, 1),
        max_consecutive_labels: int = -1,
        max_increment: int = 0.05,
        distribution_name: str = "random",
        alfa: int = None,
        beta: int = None,
        k: int = None,
        k1: int = None,
        k2: int = None,
        sine_classification_function: int = 0,
        sine_c1: float = None,
        sine_c2: float = None,
        sine_c3: float = None,
        **kwargs
    ):
        """
        Parameters
        ----------
        first_examples_df: pandas.DataFrame, default: None.
            pandas DataFrame containing first examples. If None the first example will be sampled.
        generator: str, default: 'sine'.
            Classifier to use. Possible values: {
                'rt': river.synth.RandomTree,
                'sine': river.synth.Sine
            }
        generator_obj: river.synth, default: None.
            The river.synth object to use. If None a new Object will be created.
        features_range: tuple, default: (0,1)
            Tuple (min_value, max_value) of float representing the features values' range.
        max_consecutive_labels: int, default: -1.
            In case of sine generator, the maximum number of consecutive labels in the generated examples.
            If None it will be ignored.
        max_increment: float, default: 0.05.
            Max width of the random walk for each feature.
        distribution_name: str, default: 'random'
            The distribution to sample from during the examples' generation.
            If 'random' a random distribution is sampled from the possible distributions.
            Possible values: {'random', 'uniform', 'normal', 'beta', 'f', 'chi2'}
        alfa: int, default: None.
            The alfa parameter in case of distribution is beta. If None a random value is sampled.
        beta: int, default: None.
            The beta parameter in case of distribution is beta. If None a random value is sampled.
        k: int, default: None.
            The k parameter in case of distribution is chi2. If None a random value in [1,50] is sampled.
        k1: int, default: None.
            The k1 parameter in case of distribution is f. If None a random value in [1,50] is sampled.
        k2: int, default: None.
            The k2 parameter in case of distribution is f. If None a random value in [1,50] is sampled.
        sine_classification_function: int, default: 0.
            In case of generator=="sine", the Sine generator's classification function.
        sine_c1: float, default: 0 (classification_function in [0, 1]), 0.5 (classification_function in [2, 3])
            In case of classification_function==0 or 1, the value of the first coefficient in the function:
            x1 - sine_c1 - sine_c2 * np.sin(sine_c3 * x2) = 0
            In case of classification_function==2 or 3, the value of the first coefficient in the function:
            x1 - sine_c1 - sine_c2 * np.sin(sine_c3 * np.pi * x2) = 0
        sine_c2: float, default: 1 (classification_function in [0, 1]), 0.3 (classification_function in [2, 3])
            In case of classification_function==0 or 1, the value of the second coefficient in the function:
            x1 - sine_c1 - sine_c2 * np.sin(sine_c3 * x2) = 0
            In case of classification_function==2 or 3, the value of the second coefficient in the function:
            x1 - sine_c1 - sine_c2 * np.sin(sine_c3 * np.pi * x2) = 0
        sine_c3: float, default: 1 (classification_function in [0, 1]), 3 (classification_function in [2, 3])
            In case of classification_function==0 or 1, the value of the third coefficient in the function:
            x1 - sine_c1 - sine_c2 * np.sin(sine_c3 * x2) = 0
            In case of classification_function==2 or 3, the value of the third coefficient in the function:
            x1 - sine_c1 - sine_c2 * np.sin(sine_c3 * np.pi * x2) = 0
        kwargs: dict, default: {}.
            The synth.RandomTree generator parameters, if generator_obj is None and generator = 'rt'.
            If not provided, default values are used.
        """
        self.distribution_name = None
        self.k1 = None
        self.k2 = None
        self.k = None
        self.alfa = None
        self.beta = None
        self.distribution = None
        self.data = first_examples_df
        self.train_data = None
        self.test_data = None
        self.features_range = features_range
        self.max_increment = max_increment
        self.generator = generator
        self.max_consecutive_labels = max_consecutive_labels
        self.actual_max_consecutive_labels = self.max_consecutive_labels

        self.sine_classification_function = sine_classification_function
        self.sine_c1 = sine_c1
        self.sine_c2 = sine_c2
        self.sine_c3 = sine_c3
        if self.sine_c1 is None:
            if self.sine_classification_function in [0, 1]:
                self.sine_c1 = 0
            else:
                self.sine_c1 = 0.5
        if self.sine_c2 is None:
            if self.sine_classification_function in [0, 1]:
                self.sine_c2 = 1
            else:
                self.sine_c2 = 0.3
        if self.sine_c3 is None:
            if self.sine_classification_function in [0, 1]:
                self.sine_c3 = 1
            else:
                self.sine_c3 = 3

        self.change_distribution(distribution_name, alfa, beta, k, k1, k2)
        if self.data is not None:
            self.class_col = self.data.columns[-1]
        else:
            self.class_col = "target"

        if generator == "rt":
            if generator_obj is None:
                if "n_num_features" not in kwargs:
                    kwargs["n_num_features"] = 5
                if "n_cat_features" not in kwargs:
                    kwargs["n_cat_features"] = 0
                if "n_classes" not in kwargs:
                    kwargs["n_classes"] = 2
                self.generator_obj = synth.RandomTree(**kwargs)
                self.n_features = 5
            else:
                self.n_features = generator_obj.n_num_features
                self.generator_obj = generator_obj
            self.classify = self._classify_rt
            self.max_consecutive_labels = -1
        elif generator == "sine":
            self._sign = {}
            self.n_features = 2
            if self.sine_classification_function == 0:
                self._sine_func = self._sine_01
                self.classify = self._classify_sine0
                self._sign[1] = +1
                self._sign[0] = -1
            elif self.sine_classification_function == 1:
                self._sine_func = self._sine_01
                self.classify = self._classify_sine1
                self._sign[1] = -1
                self._sign[0] = +1
            elif self.sine_classification_function == 2:
                self._sine_func = self._sin_23
                self.classify = self._classify_sine2
                self._sign[1] = +1
                self._sign[0] = -1
            else:
                self._sine_func = self._sin_23
                self.classify = self._classify_sine3
                self._sign[1] = -1
                self._sign[0] = +1

    def change_distribution(
        self,
        distribution_name: str = "random",
        alfa: int = None,
        beta: int = None,
        k: int = None,
        k1: int = None,
        k2: int = None,
    ):
        """
        Change the random walk's distribution.

        Parameters
        ----------
        distribution_name: str, default: 'random'
            The distribution to sample from during the examples' generation.
            If 'random' a random distribution is sampled from the possible distributions.
            Possible values: {'random', 'uniform', 'normal', 'beta', 'f', 'chi2'}
        alfa: int, default: None.
            The alfa parameter in case of distribution is beta. If None a random value is sampled.
        beta: int, default: None.
            The beta parameter in case of distribution is beta. If None a random value is sampled.
        k: int, default: None.
            The k parameter in case of distribution is chi2. If None a random value in [1,50] is sampled.
        k1: int, default: None.
            The k1 parameter in case of distribution is f. If None a random value in [1,50] is sampled.
        k2: int, default: None.
            The k2 parameter in case of distribution is f. If None a random value in [1,50] is sampled.
        Returns
        -------

        """
        self.distribution_name = distribution_name
        self.alfa = alfa
        self.beta = beta
        self.k = k
        self.k1 = k1
        self.k2 = k2
        if self.distribution_name == "random":
            distributions = ['uniform', 'normal', 'beta', 'f', 'chi2']
            self.distribution_name = distributions[np.random.randint(0, len(distributions))]
        if self.distribution_name == "uniform":
            self.distribution = self._sample_uniform
        elif self.distribution_name == "normal":
            self.distribution = self._sample_normal
        elif self.distribution_name == "beta":
            if self.alfa is None:
                self.alfa = np.random.random() * 5
            if self.beta is None:
                self.beta = np.random.random() * 5
            self.distribution = self._sample_beta
        elif self.distribution_name == "chi2":
            if self.k is None:
                self.k = np.random.randint(1, 51)
            self.distribution = self._sample_chi2
        elif self.distribution_name == "f":
            if self.k1 is None:
                self.k1 = np.random.randint(1, 51)
            if self.k2 is None:
                self.k2 = np.random.randint(1, 51)
            self.distribution = self._sample_f

    def _classify_rt(self, x):
        return self.generator_obj._classify_instance(self.generator_obj.tree_root, x)

    def _sine_01(self, x):
        return self.sine_c1 + self.sine_c2 * np.sin(self.sine_c3 * x[1])

    def _sin_23(self, x):
        return self.sine_c1 + self.sine_c2 * np.sin(self.sine_c3 * np.pi * x[1])

    def _classify_sine0(self, x):
        return 0 if (x[0] - self._sine_01(x)) >= 0 else 1

    def _classify_sine1(self, x):
        return 0 if (x[0] - self._sine_01(x)) < 0 else 1

    def _classify_sine2(self, x):
        return 0 if (x[0] - self._sin_23(x) >= 0) else 1

    def _classify_sine3(self, x):
        return 0 if (x[0] - self._sin_23(x)) < 0 else 1

    def _change_label (self, value):
        x_new = self.data.iloc[-1, :-1].to_dict()
        x_new[self.data.columns[0]] = (
                self._sine_func(x_new)
                + self._sign[value] * 2 * self.max_increment
                + self._sign[value] * self.distribution() * self.max_increment
        )
        self.actual_max_consecutive_labels = np.random.randint(
            int(self.max_consecutive_labels / 5), self.max_consecutive_labels + 1
        )
        return x_new

    def _generate_new_example(
        self
    ):
        x_new = {}
        idx = 1 if self.actual_max_consecutive_labels == -1 else self.actual_max_consecutive_labels
        vc = self.data[self.class_col].iloc[-idx:].value_counts().reset_index()
        value = vc.iloc[0, 0]
        count = vc.iloc[0, 1]
        if count == self.actual_max_consecutive_labels:
            x_new = self._change_label(value)
        else:
            for c in [col for col in self.data.columns if col != self.class_col]:
                if self.data[c].iloc[-1] + self.max_increment >= self.features_range[1]:
                    sign = -1
                elif self.data[c].iloc[-1] - self.max_increment <= self.features_range[0]:
                    sign = 1
                else:
                    sign = [-1, 1][np.random.randint(0, 2)]
                x_new[c] = self.data[c].iloc[-1] + sign * self.distribution() * self.max_increment
        x_new[self.class_col] = self.classify(np.array([x_new[k] for k in x_new]))
        self.data = pd.concat([self.data, pd.DataFrame.from_dict({k: [v] for k, v in x_new.items()})])

    def _sample_normal(self):
        x = np.random.normal()
        return scipy.stats.norm().cdf(x)

    def _sample_uniform(self):
        return np.random.random()

    def _sample_beta(self):
        x = np.random.beta(self.alfa, self.beta)
        return scipy.stats.beta(self.alfa, self.beta).cdf(x)

    def _sample_chi2(self):
        x = np.random.chisquare(self.k)
        return scipy.stats.chi2(self.k).cdf(x)

    def _sample_f(self):
        x = np.random.f(self.k1, self.k2)
        return scipy.stats.f(self.k1, self.k2).cdf(x)

    def _add_target_dependencies(self, temporal_order=5, n_start=0, col_label="target"):
        new_targets = list(self.data[col_label].iloc[:n_start+1])
        if n_start == 0:
            n_start = 1
        for i in range(n_start, len(self.data)):
            new_targets.append(
                st.mode(
                    self.data.iloc[
                        max(0, i - temporal_order + 1):i + 1
                    ][col_label]
                )
            )
        self.data = self.data.rename(columns={"target": "classification"})
        self.data["target"] = new_targets

    def generate(
        self,
        n_examples=100000,
        train_test_sep=None,
        target_dependencies=None,
    ):
        """
        It generates n_examples examples.
        The first example (if not provided in the class constructor method) is generated using the synth generator.
        The following examples are generated by implementing a random walk. For each feature, the i-th example is
        generated by summing/subtracting to the (i-1)-th example's value a random walk sampled from a specific
        distribution. After this random walk, each feature cannot exceed the provided features_range (0;1 if not
        provided).
        In case of sine generator, if the sequence reaches max_consecutive_labels consecutive equal labels, then the
        label value is changed.
        All the examples are classified using the provided classifier.

        Parameters
        ----------
        n_examples: int, default: 100000
            Number of examples to generate.
        train_test_sep: int, default: None
            If not None, it represents the timestamp to use as train/test separator.
        target_dependencies: int, default: None
            If not None, the target of the i-th examples is the most present RandomTree's classification within the
            previous (i-1)-th examples and the current one.
        Returns
        -------
        If train_test_sep is None it returns a pandas DataFrame containing the generated dataset.
        Otherwise, it returns a tuple of pandas DataFrames representing the train and the test datasets.
        """
        if self.data is None:
            x = {
                f"x{i}": [np.random.uniform(self.features_range[0], self.features_range[1])]
                for i in range(1, self.n_features+1)
            }
            x[self.class_col] = [self.classify(np.array([x[k] for k in x]))]
            self.data = pd.DataFrame.from_dict(x)
            start = 1
            n_start = 0
        else:
            start = 0
            n_start = len(self.data)

        if self.max_consecutive_labels != -1:
            self.actual_max_consecutive_labels = np.random.randint(1, self.max_consecutive_labels + 1)
        else:
            self.actual_max_consecutive_labels = -1

        for i in range(start, n_examples):
            self._generate_new_example()
            if (i+1) % 5 == 0:
                print(i + 1, "/", n_examples, end="\r")
        print()

        if target_dependencies is not None:
            self._add_target_dependencies(target_dependencies, n_start)

        if train_test_sep is None:
            return self.data
        else:
            self.train_data = self.data.iloc[0:train_test_sep, 0:]
            self.test_data = self.data.iloc[train_test_sep:, 0:]
            return self.train_data, self.test_data
