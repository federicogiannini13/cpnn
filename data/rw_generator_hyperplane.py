from data.rw_generator import RandomWalkGenerator
import numpy as np


class RandomWalkGeneratorHyperplane(RandomWalkGenerator):
    def __init__(
        self,
        n_features=10,
        change_proportion=None,
        coefficients=None,
        swap_classes=True,
        **kwargs
    ):
        super().__init__(generator="hyperplane", **kwargs)
        self.n_features = n_features
        if coefficients is None:
            self.coefficients = self._generate_line()
        else:
            self.coefficients = coefficients.copy()
            if change_proportion is not None:
                self.change_proportion = change_proportion
                self._change_coeff()
        self.classify = self._classify_hp
        if swap_classes:
            self.class_sign = -1
        else:
            self.class_sign = 1

    def _generate_line(self):
        coeff = []
        for i in range(self.n_features - 1):
            max_value = (1 - np.sum(coeff)) / (self.n_features - len(coeff))
            max_value += 0.2 * max_value
            coeff.append(np.random.uniform(0, max_value))
        max_value = 1 - np.sum(coeff)
        coeff.append(np.random.uniform(0, max_value))
        return coeff

    def _change_coeff(self):
        n_dim = len(self.coefficients)
        n_coef = int(len(self.coefficients) * self.change_proportion)
        if n_coef > 0:
            coeff_to_change = sorted(
                np.random.choice(
                    list(range(len(self.coefficients))), size=n_coef, replace=False
                )
            )
            coeff_to_keep = [
                i for i in range(len(self.coefficients)) if i not in coeff_to_change
            ]
            coeff_sum = np.sum(np.take(self.coefficients, coeff_to_keep))
            n_coeff_ok = len(coeff_to_keep)
            for i in coeff_to_change[:-1]:
                max_value = (1 - coeff_sum) / (n_dim - n_coeff_ok)
                max_value += 0.2 * max_value
                self.coefficients[i] = np.random.uniform(0, max_value)
                n_coeff_ok += 1
                coeff_sum += self.coefficients[i]
            i = coeff_to_change[-1]
            max_value = 1 - coeff_sum
            self.coefficients[i] = np.random.uniform(0, max_value)

    def _classify_hp(self, x):
        value = 0
        for i in range(len(x) - 1):
            value += self.coefficients[i] * x[i]
        value += self.coefficients[-1]
        value = x[-1] - value
        return 1 if self.class_sign * value > 0 else 0

    def _change_label(self, value):
        x_new = self.data.iloc[-1, :-1].to_dict()
        z_old = x_new[self.data.columns[-2]]
        z_new = (
            np.sum(
                [
                    self.coefficients[i] * x_new[self.data.columns[i]]
                    for i in range(len(self.coefficients) - 1)
                ]
            )
            + self.coefficients[-1]
        )
        if z_new > z_old:
            sign = +1
        else:
            sign = -1
        x_new[self.data.columns[-2]] = (
            z_new
            + sign * 2 * self.max_increment
            + sign * self.distribution() * self.max_increment
        )
        self.actual_max_consecutive_labels = np.random.randint(
            int(self.max_consecutive_labels / 5), self.max_consecutive_labels + 1
        )
        return x_new
