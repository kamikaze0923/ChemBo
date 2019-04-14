"""
  Unit tests for distance computation.
  -- kirthevasank
"""

from dragonfly.utils.base_test_class import BaseTestClass, execute_tests
# Local imports
from mols.molecule import Molecule


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

  def setUp(self):
    """ Set up. """
    self.data_file_names = ['data_prop.txt']

  def test_distance_computation(self):
    """ Test for distance computation. """
    for data_file_name in self.data_file_names:
      smile_strings = get_smile_strings_from_file(data_file_name)
      print(smile_strings)
      chem_molecules = [Molecule(elem) for elem in smile_strings]
      print(chem_molecules)
      mol0 = chem_molecules[0]
      print(type(mol0))
      import pdb; pdb.set_trace()
    


if __name__=="__main__":
    execute_tests()

