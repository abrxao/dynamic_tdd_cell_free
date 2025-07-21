# This module control my scenario and entities. A class of an square grid to place
# entities(Users, BSs, etc) and a class to control the entities.

# All the values of power are in dBm and the positions are in meters.
# The data rate is in Mbps.

from typing import List, Dict, Any
import tensorflow as tf


class SquareGrid:
    """
    A square grid to place entities like users and base stations. The users and
    base stations are placed in a grid of size width x height. The positions are
    generated according to the grid size, the number of entities, the type of entity,
    and the distribution of positions.
    """

    def __init__(self, width: int, height: int):
        self.width = width  # in meters
        self.height = height  # in meters

    @property
    def width(self) -> int:
        return self._width

    @width.setter
    def width(self, value: int):
        if value <= 0:
            raise ValueError("Width must be a positive integer.")
        self._width = value

    @property
    def height(self) -> int:
        return self._height

    @height.setter
    def height(self, value: int):
        if value <= 0:
            raise ValueError("Height must be a positive integer.")
        self._height = value
