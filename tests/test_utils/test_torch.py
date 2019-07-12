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

        self.assertListEqual(sorted(["a", "b", "_c"]), sorted(x.get_attribute_names()))
