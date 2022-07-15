import numpy as np

# Constants
gravity_accel = -9.81 # m/s^2

class KalmanFilter:
    def __init__(self, dt: float, init_x_R = float,
        init_x_b_c = float, init_y_b_c = float, init_z_b_c = float,
        init_d_z_b_c = float, init_v_x_c = float, 
        init_v_y_c = float, init_v_x_err = float,
        init_v_y_err = float, init_g_err = float):
        """
        eg. x_b = x_b_c - x_R where x_b_c is the ball's position in the camera frame
        and x_R is the camera's position based on forward kinematics
        """
        self.DIM = 4
        # Constants based on `dt`
        self.A = self.init_A(dt)
        self.B = self.init_B(dt)
        self.G = self.init_G(dt)
        self.sigma = self.init_sigma(init_v_x_err, init_v_y_err, init_g_err)
        self.Q = self.get_G().dot(self.get_G().T).dot(self.get_sigma()**2)

        # Initialize values based on measurements
        self.curr_x = self.set_X(
            init_x_b_c - init_x_R,
            init_y_b_c - init_x_R,
            init_z_b_c - init_x_R,
            init_d_z_b_c - init_x_R,
        )
        self.curr_u = self.set_u(
            init_v_x_c - init_x_R,
            init_v_y_c - init_x_R
        ) 
        self.curr_P = np.eye(self.DIM)

    def get_A(self):
        return self.A

    def get_B(self):
        return self.B

    def get_G(self):
        return self.G

    def get_Q(self):
        return self.Q

    def get_sigma(self):
        return self.sigma

    def get_curr_x(self):
        return self.curr_x

    def get_curr_u(self):
        return self.curr_u

    def get_curr_P(self):
        return self.curr_P

    def init_A(self, dt):
        A = np.eye(self.DIM)
        A[2][3] = dt
        return A 

    def init_B(self, dt):
        B = np.diag(np.full(self.DIM, dt))
        B[2][2] = (1/2)*(dt**2)
        return B

    def init_G(self, dt):
        G = np.full(self.DIM, dt)
        G[2] = (1/2)*(dt**2)
        return G

    def init_sigma(self, v_x_err, v_y_err, g_err):
        sigma = np.zeros(self.DIM)
        sigma[0][0] = v_x_err
        sigma[1][1] = v_y_err
        sigma[2][2] = g_err
        sigma[3][3] = g_err

    def generate_X(self, x_b, y_b, z_b, d_z_b):
        x = np.zeros(self.DIM)
        x[0] = x_b
        x[1] = y_b
        x[2] = z_b
        x[3] = d_z_b
        return x

    def generate_u(self, v_x, v_y):
        x = np.zeros(self.DIM)
        x[0] = v_x
        x[1] = v_y
        x[2] = gravity_accel
        x[3] = gravity_accel
        return x

    def generate_R(self, x_b_err, y_v_err, z_b_err, d_z_b_err):
        R = np.eye(self.DIM)
        R[0][0] = x_b_err
        R[1][1] = y_v_err
        R[2][2] = z_b_err
        R[3][3] = d_z_b_err
        return R

    def predict(self):
        new_x = self.get_A().dot(self.get_curr_x()) + B.dot(self.get_curr_u())
        new_P = self.get_A().dot(self.get_curr_P()).dot(self.get_A().T) + self.get_Q()

        # Assign predicted values
        self.curr_x = new_x
        self.curr_P = new_P

    def update(self, x_b_err, y_v_err, z_b_err, d_z_b_err):
        R = self.generate_R(x_b_err, y_v_err, z_b_err, d_z_b_err)
        # y = z - Hx
        H = np.eye(self.DIM)
        S = H.dot(self.get_curr_P()).dot(H.T) + R
        k = self.get_curr_P().dot(H.T).dot(np.linalg.inv(S))

        # TODO: Populate w/ measurements in `z`
        updated_x = self.get_curr_x() + k.dot(z - H.dot(self.get_curr_x()))
        updated_P = (np.eye(self.DIM) - k.dot(H)).dot(self.get_curr_P())

        predict_err = abs(self.get_curr_x() - updated_x)
        # TODO: Replace w/ setters
        self.curr_x = updated_x
        self.curr_P = updated_P

    def run(self, dt):
        # TODO: Put run and update w/ timesteps
        pass
