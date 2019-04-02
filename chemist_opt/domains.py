"""
A child Cartesian product domain
that works with molecules

Or does it suffice to only redefine utility?
(from cp_domain_utils.py)

"""

# TODO: needed imports
# from dragonfly.exp.cp_domain_utils import *


def load_cp_domain_from_config_file(config_file, *args, **kwargs):
    """ Loads and creates a cartesian product domain object from a config_file. """
    parsed_result = config_parser(config_file)
    domain_params = parsed_result['domain']
    domain_constraints = None if not ('domain_constraints' in parsed_result.keys()) \
                        else parsed_result['domain_constraints']
    domain_info = Namespace(config_file=config_file)
    return load_domain_from_params(domain_params, domain_constraints=domain_constraints,
                                   domain_info=domain_info, *args, **kwargs)


#------------------------------------------------------------------------------
# Maybe even just copy all of the stuff
# from exp.cp_domain_utils and add mol support
#------------------------------------------------------------------------------

def load_domain_from_params(domain_params,
        domain_constraints=None, domain_info=None):
    """ Loads and creates a cartesian product object from a config_file.
        Supports adding MolDomain.
    """
    pass

