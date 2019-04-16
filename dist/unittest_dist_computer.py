"""
  Unit tests for distance computation.
  -- kirthevasank
"""

# pylint: disable=import-error
# pylint: disable=invalid-name

from dragonfly.utils.base_test_class import BaseTestClass, execute_tests
# Local imports
from dist.qp_dist_computer import QPChemDistanceComputer


def get_smile_strings_from_file(data_file_name):
  """ Returns the chemical molecules. """
  data_file = open(data_file_name)
  lines = data_file.readlines()
  stripped_lines = [elem.strip() for elem in lines]
  words = [elem.split() for elem in stripped_lines]
  smile_strings = [elem[0] for elem in words if len(elem) > 0]
  return smile_strings


class DistComputerTestCases(BaseTestClass):
  """ Unit tests for distance computation. """
  # pylint: disable=no-init

  def setUp(self):
    """ Set up. """
    self.data_file_names = ['data_alkanes.txt', 'data_prop.txt']
#     self.data_file_names = ['data_prop.txt']
    self.qp_dist_computer = QPChemDistanceComputer(
        [0.5, 1.0, 2.0], non_assignment_penalty=1.0,
        nonexist_non_assignment_penalty_vals=[3.0, 6.0])

  def test_distance_computation(self):
    """ Test for distance computation. """
    for data_file_name in self.data_file_names:
      smile_strings = get_smile_strings_from_file(data_file_name)
      distances = self.qp_dist_computer(smile_strings, smile_strings)
      print(distances)
      import pdb; pdb.set_trace()



if __name__ == "__main__":
  execute_tests()

