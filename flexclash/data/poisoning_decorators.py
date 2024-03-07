"""
Copyright (C) 2024  Instituto Andaluz Interuniversitario en Ciencia de Datos e Inteligencia Computacional (DaSCI)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import functools

from flex.common.utils import check_min_arguments
from flex.data import Dataset

from flexclash.data.dataset import PoisonedDataset


def data_poisoner(func):
    min_args = 2
    assert check_min_arguments(
        func, min_args
    ), f"The decorated function: {func.__name__} is expected to have at least {min_args} argument/s."

    @functools.wraps(func)
    def _poison_Dataset_(
        client_dataset: Dataset,
        *args,
        **kwargs,
    ):
        def poison_func(label, feature):
            try:
                new_label, new_feature = func(label, feature, *args, **kwargs)
            except ValueError:
                raise ValueError(
                    "The decorated function: {func.__name__} must return two values: features, labels."
                )
            return new_label, new_feature

        if isinstance(client_dataset, PoisonedDataset):

            def new_poison_func(label, feature):
                return poison_func(*client_dataset.poisoning_function(label, feature))

            poisoned_dataset = PoisonedDataset(
                X_data=client_dataset.X_data,
                y_data=client_dataset.y_data,
                poisoning_function=new_poison_func,
            )
        else:
            poisoned_dataset = PoisonedDataset(
                X_data=client_dataset.X_data,
                y_data=client_dataset.y_data,
                poisoning_function=poison_func,
            )
        return poisoned_dataset

    return _poison_Dataset_
