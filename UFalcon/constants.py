from astropy import constants

"""
This file contains the physical constants in the right units
used by the UFalcon package
"""

# speed of light
c = constants.c.to("km / s").value