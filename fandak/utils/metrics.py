from collections import defaultdict
from numbers import Number
from pathlib import Path
from pickle import dump
from typing import Optional

from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from fandak.utils.misc import print_with_time
from fandak.utils.torch import GeneralDataClass


def is_scalar_like(ds: GeneralDataClass, an: str) -> bool:
    a = getattr(ds, an)
    if isinstance(a, Tensor):  # is Tensor
        try:  # has .item()
            a.item()
        except ValueError:
            return False
        return True
    elif isinstance(a, Number):  # is Number
        return True
    else:
        return False


class ScalarMetricCollection:
    def __init__(
        self,
        writer: Optional[SummaryWriter],
        root: Path,
        base_name: str,
        print_each_iter: bool = False,
        report_average: bool = True,
    ):
        self.writer = writer
        self.root = root
        self.base_name = base_name
        self.print_each_iter = print_each_iter
        self.report_average = report_average
        self.values = defaultdict(list)
        self.average_base_tag = f"training_average/{self.base_name}"

    def add_value(self, dc_value: GeneralDataClass, step: int):
        loss_like_attr_names = dc_value.filter_attributes(
            is_scalar_like, initial_attr_list=dc_value.get_attribute_names()
        )

        for attr_name in loss_like_attr_names:
            tag_name = f"{self.base_name}/{attr_name}"
            attr = getattr(dc_value, attr_name)
            if isinstance(attr, Tensor):
                value = attr.item()
            else:
                value = attr
            if self.writer:
                self.writer.add_scalar(tag_name, scalar_value=value, global_step=step)
            self.values[attr_name].append(value)
            if self.print_each_iter:
                print_with_time(f"(step {step}) {tag_name}: {value}")

    def set_value(self, dc_value: GeneralDataClass, step: int):
        loss_like_attr_names = dc_value.filter_attributes(
            is_scalar_like, initial_attr_list=dc_value.get_attribute_names()
        )

        for attr_name in loss_like_attr_names:
            attr = getattr(dc_value, attr_name)
            if isinstance(attr, Tensor):
                value = attr.item()
            else:
                value = attr
            try:
                self.values[attr_name][step] = value
            except (KeyError, IndexError):
                pass

    def epoch_finished(self, epoch_num: int):
        if self.report_average:
            average_values = {}
            for attr_name in self.values.keys():
                average_value = self.average_value(attr_name)
                average_values[attr_name] = average_value
                tag_name = f"{self.average_base_tag}/{attr_name}"
                if self.writer:
                    self.writer.add_scalar(
                        tag=tag_name,
                        scalar_value=average_value,
                        global_step=epoch_num + 1,
                    )
                print_with_time(f"{tag_name}: {average_value}")
            self.save(
                dictionary=average_values,
                name=Path(str(epoch_num + 1)) / self.base_name,
            )
        self.reset_values()

    def reset_values(self):
        for k in self.values.keys():
            self.values[k].clear()

    def average_value(self, attr_name: str) -> float:
        return sum(self.values[attr_name]) / len(self.values[attr_name])

    def save(self, dictionary: dict = None, name: str = None):
        if dictionary is None:
            dictionary = self.values
        if name is None:
            name = f"{self.base_name}"
        file_path = self.root / f"{name}.pkl"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as f:
            dump(dictionary, f)
