from collections import defaultdict
from numbers import Number

from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from fandak.utils import print_with_time
from fandak.utils.torch import GeneralDataClass


class ScalarMetric:
    # TODO: refactor to inherit ScalarMetricCollection
    def __init__(self, writer: SummaryWriter, name: str, report_average: bool = True):
        self.writer = writer
        self.name = name
        self.report_average = report_average
        self.values = []
        self.average_tag = "training_average/%s" % self.name

    def add_value(self, value: float, step: int):
        self.writer.add_scalar(tag=self.name, scalar_value=value, global_step=step)
        self.values.append(value)

    def epoch_finished(self, epoch_num: int):
        average_value = self.average_value()
        if self.report_average:
            self.writer.add_scalar(
                tag=self.average_tag,
                scalar_value=average_value,
                global_step=epoch_num + 1,
            )
            print_with_time("%s: %f" % (self.average_tag, average_value))
        self.reset_values()

    def reset_values(self):
        self.values.clear()

    def average_value(self) -> float:
        return sum(self.values) / len(self.values)


class ScalarMetricCollection:
    def __init__(
        self, writer: SummaryWriter, base_name: str, report_average: bool = True
    ):
        self.writer = writer
        self.base_name = base_name
        self.report_average = report_average
        self.values = defaultdict(list)
        self.average_base_tag = "training_average/%s" % self.base_name

    def add_value(self, dc_value: GeneralDataClass, step: int):
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

        loss_like_attr_names = dc_value.filter_attributes(
            is_scalar_like, initial_attr_list=dc_value.get_attribute_names()
        )

        for attr_name in loss_like_attr_names:
            tag_name = "{base_name}/{attr_name}".format(
                base_name=self.base_name, attr_name=attr_name
            )
            attr = getattr(dc_value, attr_name)
            if isinstance(attr, Tensor):
                value = attr.item()
            else:
                value = attr
            self.writer.add_scalar(tag_name, scalar_value=value, global_step=step)
            self.values[attr_name].append(value)

    def epoch_finished(self, epoch_num: int):
        if self.report_average:
            for attr_name in self.values.keys():
                average_value = self.average_value(attr_name)
                tag_name = "{base_name}/{attr_name}".format(
                    base_name=self.average_base_tag, attr_name=attr_name
                )
                self.writer.add_scalar(
                    tag=tag_name, scalar_value=average_value, global_step=epoch_num + 1
                )
                print_with_time("%s: %f" % (tag_name, average_value))
        self.reset_values()

    def reset_values(self):
        for k in self.values.keys():
            self.values[k].clear()

    def average_value(self, attr_name: str) -> float:
        return sum(self.values[attr_name]) / len(self.values[attr_name])
