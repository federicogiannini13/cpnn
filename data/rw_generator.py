import scipy
import scipy.stats
from river import synth
import pandas as pd
import numpy as np


class RandomWalkGenerator:
    def __init__(
        self, first_examples_df=None, generator="rt", generator_obj=None, **kwargs
    ):
        """
        Parameters
        ----------
        first_examples_df: pandas.DataFrame, default: None
            pandas DataFrame containing first examples. If None the first example will be sampled.
        generator: str, default: 'rt'
            Classifier to use. Possible values: {
                'rt': river.synth.RandomTree,
                'sine': river.synth.Sine
            }
        generator_obj: river.synth, default: None
            The river.synth object to use. If None a new Object will be created.
        kwargs:
            river.synth Generator parameters.
        """
        self.data = first_examples_df
        self.alfa = None
        self.beta = None
        self.k = None
        self.k1 = None
        self.k2 = None
        self.generator = generator

        if generator == "rt":
            if "n_num_features" not in kwargs:
                kwargs["n_num_features"] = 5
            if "n_cat_features" not in kwargs:
                kwargs["n_cat_features"] = 0
            if "n_classes" not in kwargs:
                kwargs["n_classes"] = 2
            if generator_obj is None:
                self.generator_obj = synth.RandomTree(**kwargs)
            else:
                self.generator_obj = generator_obj
            self.classify = self._classify_rt
        else:
            self._sign = {}
            if "classification_function" not in kwargs:
                kwargs["classification_function"] = 0
            if kwargs["classification_function"] == 0:
                self._sine_func = self._sin_0_1
                self.classify = self._classify_sin_0
                self._sign[1] = +1
                self._sign[0] = -1
            elif kwargs["classification_function"] == 1:
                self._sine_func = self._sin_0_1
                self.classify = self._classify_sin_1
                self._sign[1] = -1
                self._sign[0] = +1
            elif kwargs["classification_function"] == 2:
                self._sine_func = self._sin_2_3
                self.classify = self._classify_sin_2
                self._sign[1] = +1
                self._sign[0] = -1
            else:
                self._sine_func = self._sin_2_3
                self.classify = self._classify_sin_3
                self._sign[1] = -1
                self._sign[0] = +1
            self.generator_obj = synth.Sine(**kwargs)

    def _classify_rt(self, x):
        return self.generator_obj._classify_instance(self.generator_obj.tree_root, x)

    def _sin_0_1(self, x):
        return np.sin(x[1])

    def _sin_2_3(self, x):
        return 0.5 + 0.3 * np.sin(3 * np.pi * x[1])

    def _classify_sin_0(self, x):
        return 0 if (x[0] - self._sin_0_1(x)) >= 0 else 1

    def _classify_sin_1(self, x):
        return 0 if (x[0] - self._sin_0_1(x)) < 0 else 1

    def _classify_sin_2(self, x):
        return 0 if (x[0] - self._sin_2_3(x) >= 0) else 1

    def _classify_sin_3(self, x):
        return 0 if (x[0] - self._sin_2_3(x)) < 0 else 1

    def _generate_new_example(
        self, df, sampling_func, max_consec_labels=0, class_col="target"
    ):
        x_new = {}
        idx = 1 if self.actual_max_consec_labels == 0 else self.actual_max_consec_labels
        vc = df[class_col].iloc[-idx:].value_counts().reset_index()
        value = vc.iloc[0, 0]
        count = vc.iloc[0, 1]
        if count == self.actual_max_consec_labels:
            x_new = df.iloc[-1, :-1].to_dict()
            x_new[df.columns[0]] = (
                self._sine_func(x_new)
                + self._sign[value] * 2 * self.max_increment
                + self._sign[value] * sampling_func() * self.max_increment
            )
            self.actual_max_consec_labels = np.random.randint(
                int(max_consec_labels / 5), max_consec_labels + 1
            )
        else:
            for c in [col for col in df.columns if col != class_col]:
                if df[c].iloc[-1] + self.max_increment >= self.features_range[1]:
                    sign = -1
                elif df[c].iloc[-1] - self.max_increment <= self.features_range[0]:
                    sign = 1
                else:
                    sign = [-1, 1][np.random.randint(0, 2)]
                x_new[c] = df[c].iloc[-1] + sign * sampling_func() * self.max_increment
        x_new[class_col] = self.classify(x_new)
        df = pd.concat([df, pd.DataFrame.from_dict({k: [v] for k, v in x_new.items()})])
        return df

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

    def _add_target_dependencies(self, df, n, col_label="target"):
        new_targets = list(df[col_label].iloc[:n])
        for i in range(n, len(df)):
            new_targets.append(
                df.iloc[i - n + 1 : i + 1, 0:][col_label]
                .value_counts()
                .reset_index()["index"]
                .iloc[0]
            )
        df = df.rename(columns={"target": "classification"})
        df["target"] = new_targets
        return df

    def generate(
        self,
        n_examples=100000,
        max_increment=0.05,
        features_range=(0, 1),
        distribution="uniform",
        max_consec_labels=None,
        train_test_sep=None,
        target_dependencies=None,
        **kwargs
    ):
        """
        It generates n_examples examples.
        The first example (if not provided in the class constructor method) is generated using the synth generator.
        The following examples are generated by implementing a random walk. For each feature, the i-th example is
        generated by summing/subtracting to the (i-1)-th example's value a random walk sampled from a specific
        distribution. After this random walk, each feature cannot exceed the provided features_range (0;1 if not
        provided).
        In case of sine generator, if the sequence reaches max_conseq_labels consecutive equal labels, then the label
        value is changed.
        All the examples are classified using the provided classifier.

        Parameters
        ----------
        n_examples: int, default: 100000
            Number of examples to generate.
        max_increment: float, default: 0.05
            Max width of the random walk for each feature.
        features_range: tuple, default: (0,1)
            Tuple (min_value, max_value) of float representing the features values' range.
        distribution: str, default: 'uniform'
            The distribution to sample from during the examples' generation.
            Possible values: {'uniform', 'normal', 'beta', 'f', 'chi2'}
        max_consec_labels: int, default: None
            In case of sine generator, the maximum number of consecutive labels in the generated examples.
            If None it will be ignored.
        train_test_sep: int, default: None
            If not None, it represents the timestamp to use as train/test separator.
        target_dependencies: int, default: None
            If not None, the target of the i-th examples is the most present RandomTree's classification within the
            previous (i-1)-th examples and the current one.
        kwargs:
            Any parameters of the distribution (if not provided parameters are sampled from a uniform distribution).
            - k for chi2 distribution
            - k1 and k2 for f distribution
            - alfa and beta for beta distribution
        Returns
        -------
        If train_test_sep is None it returns a pandas DataFrame containing the generated dataset.
        Otherwise, it returns a tuple of pandas DataFrames representing the train and the test datasets.
        """
        if self.data is None:
            for x, y in self.generator_obj.take(1):
                pass
            for k in x:  # convert river dict to pandas from_dict format
                x[k] = [x[k]]
            x["target"] = y
            self.data = pd.DataFrame.from_dict(x)

        self.distribution_name = distribution
        self.features_range = features_range
        self.max_increment = max_increment

        if self.generator != "sine":
            max_consec_labels = 0

        sampling_func = self._sample_uniform
        if distribution == "normal":
            sampling_func = self._sample_normal
        elif distribution == "beta":
            if "alfa" not in kwargs:
                self.alfa = np.random.random() * 5
            else:
                self.alfa = kwargs["alfa"]
            if "beta" not in kwargs:
                self.beta = np.random.random() * 5
            else:
                self.beta = kwargs["beta"]
            sampling_func = self._sample_beta
        elif distribution == "chi2":
            if "k" not in kwargs:
                self.k = np.random.randint(1, 51)
            else:
                self.k = kwargs["k"]
            sampling_func = self._sample_chi2
        elif distribution == "f":
            if "k1" not in kwargs:
                self.k1 = np.random.randint(1, 51)
            else:
                self.k1 = kwargs["k1"]
            if "k2" not in kwargs:
                self.k2 = np.random.randint(1, 51)
            else:
                self.k2 = kwargs["k2"]
            sampling_func = self._sample_f

        if max_consec_labels != 0:
            self.actual_max_consec_labels = np.random.randint(1, max_consec_labels + 1)
        else:
            self.actual_max_consec_labels = 0

        for i in range(1, n_examples):
            self.data = self._generate_new_example(
                self.data, sampling_func, max_consec_labels
            )
            if (i+1)%5==0:
                print(i + 1, "/", n_examples, end="\r")
        print()

        if train_test_sep is None:
            if target_dependencies is not None:
                self.data = self._add_target_dependencies(
                    self.data, target_dependencies
                )
            return self.data
        else:
            self.train_data = self.data.iloc[0:train_test_sep, 0:]
            self.test_data = self.data.iloc[train_test_sep:, 0:]
            if target_dependencies is not None:
                self.train_data = self._add_target_dependencies(
                    self.train_data, target_dependencies
                )
                self.test_data = self._add_target_dependencies(
                    self.test_data, target_dependencies
                )
            return self.train_data, self.test_data
