from system_class import System

class Lorenz(System):
    def __init__(self) -> None:
        super().__init__()
        self.num_states = 3
        self.num_inputs = 1

    @staticmethod
    def _dynamics(x, t, u):
        """
        Lorenz Attractor dynamic system 
        (Parameters as Scipy integrate form)
        :param x (ndarray) State vector
        :param t (ndarray) Time vector
        :param u (ndarray) Input vector
        return: (list) Differential array
        """
        sigma = 10.0
        beta = 8/3
        rho = 28.0
        # Rename states (x, y, z)
        x, y, z = x
        dx = sigma * (y - x) + u
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return [dx, dy, dz]
