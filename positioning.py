import tensorflow as tf
from typing import Optional, Tuple


class Positioning:
    """
    A class to handle positioning logic for entities in a square grid.
    """

    @staticmethod
    def _handle_z_values(
        num_entities: int,
        batch_size: int,
        z_value: Optional[float] = None,
        z_min_max: Optional[Tuple[float, float]] = None,
        mean_z: float = 0.0,
        stddev_z: float = 1.0,
        distribution_type: str = "uniform",
    ) -> tf.Tensor:
        """
        Handles the generation of Z-coordinate values based on the provided parameters.

        Parameters:
        - num_entities (int): Number of entities for which to generate Z-coordinates.
        - batch_size (int): The desired batch size.
        - z_value (Optional[float]): A fixed Z-coordinate for all entities. If provided, z_min_max and mean_z/stddev_z are ignored.
        - z_min_max (Optional[Tuple[float, float]]): A tuple (min_z, max_z) for random Z-coordinate generation.
                                                       If provided, mean_z/stddev_z are ignored.
        - mean_z (float): Mean of the Gaussian distribution for Z-coordinates (used if z_value and z_min_max are None and distribution_type is 'gaussian').
        - stddev_z (float): Standard deviation of the Gaussian distribution for Z-coordinates (used if z_value and z_min_max are None and distribution_type is 'gaussian').
        - distribution_type (str): Type of distribution for Z if not fixed or min/max. Can be 'uniform' (default 0) or 'gaussian'.

        Returns:
        - tf.Tensor: A tensor with shape (batch_size, num_entities) containing the Z-coordinates.
        """
        if z_value is not None:
            return tf.fill((batch_size, num_entities), tf.cast(z_value, tf.float32))
        elif z_min_max is not None:
            min_z, max_z = z_min_max
            return tf.random.uniform(
                (batch_size, num_entities), minval=min_z, maxval=max_z, dtype=tf.float32
            )
        else:
            if distribution_type == "gaussian":
                return tf.random.normal(
                    (batch_size, num_entities),
                    mean=mean_z,
                    stddev=stddev_z,
                    dtype=tf.float32,
                )
            else:  # Default to 0 for 'uniform' or any other case if not specified
                return tf.zeros((batch_size, num_entities), dtype=tf.float32)

    @staticmethod
    def equal_distribution(
        num_entities: int,
        width: int,
        height: int,
        batch_size: int = 1,
        z_value: Optional[float] = None,
        z_min_max: Optional[Tuple[float, float]] = None,
    ) -> tf.Tensor:
        """
        Generates positions for entities using an equal distribution within the grid.

        Parameters:
        - num_entities (int): Number of entities to place.
        - width (int): Width of the grid in meters.
        - height (int): Height of the grid in meters.
        - batch_size (int): The desired batch size. Defaults to 1.
        - z_value (Optional[float]): A fixed Z-coordinate for all entities. If provided, z_min_max is ignored.
        - z_min_max (Optional[Tuple[float, float]]): A tuple (min_z, max_z) for random Z-coordinate generation.

        Returns:
        - tf.Tensor: A 3D tensor with shape (batch_size, num_entities, 3) containing the x, y, and z coordinates.
        """
        if num_entities <= 0 or width <= 0 or height <= 0:
            raise ValueError("All parameters must be positive integers.")
        # Check if the number of entities is a perfect square for equal distribution
        sqrt_n_entities = tf.math.sqrt(tf.cast(num_entities, tf.float32))
        if sqrt_n_entities - int(sqrt_n_entities) != 0:
            raise ValueError(
                "Number of entities must be a perfect square for equal distribution."
            )
        # Calculate the step size based on the grid dimensions and number of entities
        sqrt_n_entities = int(sqrt_n_entities)
        x_step = width / sqrt_n_entities
        y_step = height / sqrt_n_entities
        width -= x_step  # Adjust width to account for padding
        height -= y_step  # Adjust height to account for padding

        x_positions = tf.linspace(0.0, width, sqrt_n_entities)
        y_positions = tf.linspace(0.0, height, sqrt_n_entities)
        x_positions = tf.tile(x_positions[:, tf.newaxis], [1, sqrt_n_entities])
        y_positions = tf.tile(y_positions[tf.newaxis, :], [sqrt_n_entities, 1])
        x_positions = tf.reshape(x_positions, [-1]) + x_step / 2  # Center the positions
        y_positions = tf.reshape(y_positions, [-1]) + y_step / 2  # Center the positions

        # Handle Z-coordinate using the new function
        z_positions_flat = Positioning._handle_z_values(
            num_entities, 1, z_value, z_min_max, distribution_type="uniform"
        )
        # Reshape z_positions to match the (num_entities,) shape of x_positions/y_positions
        z_positions = tf.reshape(z_positions_flat, [-1])

        positions = tf.stack([x_positions, y_positions, z_positions], axis=-1)
        # Add batch dimension
        positions = tf.expand_dims(positions, axis=0)
        if batch_size > 1:
            positions = tf.tile(positions, [batch_size, 1, 1])

        return positions

    @staticmethod
    def uniform_distribution(
        num_entities: int,
        width: int,
        height: int,
        batch_size: int = 1,
        z_value: Optional[float] = None,
        z_min_max: Optional[Tuple[float, float]] = None,
    ) -> tf.Tensor:
        """
        Generates positions for entities using a uniform distribution within the grid.

        Parameters:
        - num_entities (int): Number of entities to place.
        - width (int): Width of the grid in meters.
        - height (int): Height of the grid in meters.
        - batch_size (int): The desired batch size. Defaults to 1.
        - z_value (Optional[float]): A fixed Z-coordinate for all entities. If provided, z_min_max is ignored.
        - z_min_max (Optional[Tuple[float, float]]): A tuple (min_z, max_z) for random Z-coordinate generation.

        Returns:
        - tf.Tensor: A 3D tensor with shape (batch_size, num_entities, 3) containing the x, y, and z coordinates.
        """
        x_positions = tf.random.uniform(
            (batch_size, num_entities), minval=0, maxval=width, dtype=tf.float32
        )
        y_positions = tf.random.uniform(
            (batch_size, num_entities), minval=0, maxval=height, dtype=tf.float32
        )

        # Handle Z-coordinate using the new function
        z_positions = Positioning._handle_z_values(
            num_entities, batch_size, z_value, z_min_max, distribution_type="uniform"
        )

        return tf.stack([x_positions, y_positions, z_positions], axis=-1)

    @staticmethod
    def gaussian_distribution(
        num_entities: int,
        width: int,
        height: int,
        mean: Tuple[float, float, float] = (0, 0, 0),
        stddev: Tuple[float, float, float] = (1, 1, 1),
        batch_size: int = 1,
        z_value: Optional[float] = None,
        z_min_max: Optional[Tuple[float, float]] = None,
    ) -> tf.Tensor:
        """
        Generates positions for entities using a Gaussian distribution within the grid.

        Parameters:
        - num_entities (int): Number of entities to place.
        - width (int): Width of the grid in meters.
        - height (int): Height of the grid in meters.
        - mean (Tuple[float, float, float]): Mean of the Gaussian distribution for x, y, and z coordinates.
        - stddev (Tuple[float, float, float]): Standard deviation of the Gaussian distribution for x, y, and z coordinates.
        - batch_size (int): The desired batch size. Defaults to 1.
        - z_value (Optional[float]): A fixed Z-coordinate for all entities. If provided, z_min_max and mean[2]/stddev[2] are ignored.
        - z_min_max (Optional[Tuple[float, float]]): A tuple (min_z, max_z) for random Z-coordinate generation. If provided, mean[2]/stddev[2] are ignored.

        Returns:
        - tf.Tensor: A 3D tensor with shape (batch_size, num_entities, 3) containing the x, y, and z coordinates.
        """
        x_positions = tf.random.normal(
            (batch_size, num_entities), mean=mean[0], stddev=stddev[0], dtype=tf.float32
        )
        y_positions = tf.random.normal(
            (batch_size, num_entities), mean=mean[1], stddev=stddev[1], dtype=tf.float32
        )

        # Handle Z-coordinate using the new function
        z_positions = Positioning._handle_z_values(
            num_entities,
            batch_size,
            z_value,
            z_min_max,
            mean_z=mean[2],
            stddev_z=stddev[2],
            distribution_type="gaussian",
        )

        # Clip positions to ensure they are within the grid boundaries
        x_positions = tf.clip_by_value(x_positions, 0, width)
        y_positions = tf.clip_by_value(y_positions, 0, height)
        # Z-clipping is handled by z_min_max if provided, otherwise assumed to be within desired range by mean/stddev

        return tf.stack([x_positions, y_positions, z_positions], axis=-1)

    @staticmethod
    def random_distribution(
        num_entities: int,
        width: int,
        height: int,
        batch_size: int = 1,
        z_value: Optional[float] = None,
        z_min_max: Optional[Tuple[float, float]] = None,
    ) -> tf.Tensor:
        """
        Generates positions for entities using a random distribution within the grid.

        Parameters:
        - num_entities (int): Number of entities to place.
        - width (int): Width of the grid in meters.
        - height (int): Height of the grid in meters.
        - batch_size (int): The desired batch size. Defaults to 1.
        - z_value (Optional[float]): A fixed Z-coordinate for all entities. If provided, z_min_max is ignored.
        - z_min_max (Optional[Tuple[float, float]]): A tuple (min_z, max_z) for random Z-coordinate generation.

        Returns:
        - tf.Tensor: A 3D tensor with shape (batch_size, num_entities, 3) containing the x, y, and z coordinates.
        """
        return Positioning.uniform_distribution(
            num_entities, width, height, batch_size, z_value, z_min_max
        )

    @staticmethod
    def get_distances(positions1: tf.Tensor, positions2: tf.Tensor) -> tf.Tensor:
        """
        Return a 3D tensor with the distances between two sets of positions.

        Parameters:
        - positions1 (tf.Tensor): A tensor with shape (batch_size, num_entities1, 3) containing the x, y, and z coordinates of the first set of positions.
        - positions2 (tf.Tensor): A tensor with shape (batch_size, num_entities2, 3) containing the x, y, and z coordinates of the second set of positions.

        Returns:
        - tf.Tensor: A 3D tensor with shape (batch_size, num_entities1, num_entities2) containing the distances.
        """
        if positions1.shape[-1] != 3 or positions2.shape[-1] != 3:
            raise ValueError("Positions must have shape (..., 3).")
        if positions1.shape[0] != positions2.shape[0]:
            raise ValueError(
                "Batch sizes of positions1 and positions2 must be the same."
            )

        # Expand dimensions for broadcasting:
        # positions1 becomes (batch_size, num_entities1, 1, 3)
        # positions2 becomes (batch_size, 1, num_entities2, 3)
        expanded_positions1 = tf.expand_dims(positions1, axis=2)
        expanded_positions2 = tf.expand_dims(positions2, axis=1)

        # Calculate the squared differences
        squared_differences = tf.square(expanded_positions1 - expanded_positions2)

        # Sum along the coordinate dimension (axis=-1) to get squared distances
        squared_distances = tf.reduce_sum(squared_differences, axis=-1)

        # Return the square root of the squared distances
        return tf.sqrt(squared_distances)
