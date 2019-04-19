"""
  Unit tests for distance computation.
  -- kirthevasank
"""

# pylint: disable=import-error
# pylint: disable=invalid-name

from dragonfly.utils.base_test_class import BaseTestClass, execute_tests
# Local imports
from dist.ot_dist_computer import OTChemDistanceComputer


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
    self.qp_dist_computer = OTChemDistanceComputer(
        non_assignment_penalty=1.0,
        nonexist_non_assignment_penalty_vals=[3.0, 6.0])

  def test_distance_computation(self):
    """ Test for distance computation. """
    self.report('Testing distance computation with dist_computer %s.'%(
                self.qp_dist_computer))
    for data_file_name in self.data_file_names:
      smile_strings = get_smile_strings_from_file(data_file_name)
      distances = self.qp_dist_computer(smile_strings, smile_strings)
      for idx, dist in enumerate(distances):
        self.report('Distances for %s using parametrisation %d:\n%s\n'%(
            data_file_name, idx, str(dist)), 'test_result')


if __name__ == "__main__":
  execute_tests()

