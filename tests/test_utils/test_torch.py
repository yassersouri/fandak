from dataclasses import dataclass
from unittest import TestCase

from fandak.utils.torch import GeneralDataClass


class TestGeneralDataClass(TestCase):
    def test_get_attribute_names(self):
        @dataclass
        class T(GeneralDataClass):
            a: int
            b: int
            _c: int

            def e(self):
                return self.a + self.b

        x = T(1, 2, 3)

        self.assertListEqual(["_c", "a", "b"], sorted(x.get_attribute_names()))

    def test_get_attribute_names_with_filter(self):
        @dataclass
        class T(GeneralDataClass):
            a: int
            aa: int
            c: int

            def e(self):
                return self.a + self.c

        x = T(1, 2, 3)

        list_of_attr = x.filter_attributes(
            lambda dc, a: True, initial_attr_list=x.get_attribute_names()
        )

        self.assertListEqual(["a", "aa", "c"], sorted(list_of_attr))

        list_of_attr = x.filter_attributes(
            lambda dc, a: a.startswith("a"), initial_attr_list=x.get_attribute_names()
        )
        self.assertListEqual(["a", "aa"], sorted(list_of_attr))
