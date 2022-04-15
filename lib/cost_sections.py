import numpy as np
import scipy.optimize as sciopt
import scipy.stats as scistats

class CostSectionGeneral:
    def __init__(self):
        self.hop_suppression_threshold = 1.e-4
        self.coeffs = np.zeros(5, dtype='f8')
        self.current = 0.
    
    def fun(self, theta=None):
        if theta is None:
            theta = self.current
            
        return self.coeffs[0] * np.sin(2. * theta) + \
                self.coeffs[1] * np.cos(2. * theta) + \
                self.coeffs[2] * np.sin(theta) + \
                self.coeffs[3] * np.cos(theta) + \
                self.coeffs[4]
    
    def grad(self, theta=None):
        if theta is None:
            theta = self.current
            
        return 2. * self.coeffs[0] * np.cos(2. * theta) + \
                (-2. * self.coeffs[1]) * np.sin(2. * theta) + \
                self.coeffs[2] * np.cos(theta) + \
                (-self.coeffs[3]) * np.sin(theta)
    
    def hess(self, theta=None):
        if theta is None:
            theta = self.current
            
        return -4. * self.coeffs[0] * np.sin(2. * theta) + \
                (-4. * self.coeffs[1]) * np.cos(2. * theta) + \
                (-self.coeffs[2]) * np.sin(theta) + \
                (-self.coeffs[3]) * np.cos(theta)
    
    def minimum(self):
        def find_next_optimum(x0, fun, grad, hess):
            return sciopt.minimize(fun, [x0], method='trust-ncg', jac=grad, hess=hess, options={'initial_trust_radius': np.pi / 8.})
        
        if self.grad(0.) < 0.:
            x0 = 0.
            shift = 0.01
        else:
            x0 = 2. * np.pi
            shift = -0.01
            
        positive_fun = lambda x: self.fun(x[0])
        positive_hess = lambda x: self.hess(x).reshape((1, 1))

        # first minimum
        res_min1 = find_next_optimum(x0, positive_fun, self.grad, positive_hess)

        negative_fun = lambda x: -self.fun(x[0])
        negative_grad = lambda x: -self.grad(x)
        negative_hess = lambda x: -self.hess(x).reshape((1, 1))
        
        # first maximum
        x0 = res_min1.x[0] + shift
        res_max1 = find_next_optimum(x0, negative_fun, negative_grad, negative_hess)
        
        # second minimum
        x0 = res_max1.x[0] + shift
        res_min2 = find_next_optimum(x0, positive_fun, self.grad, positive_hess)
        
        if (shift > 0. and res_min2.x[0] > 2. * np.pi) or (shift < 0. and res_min2.x[0] < 0.):
            return res_min1.x[0]
        elif abs(res_min1.fun - res_min2.fun) < self.hop_suppression_threshold:
            if abs(res_min1.x[0] - self.current) < abs(res_min2.x[0] - self.current):
                return res_min1.x[0]
            else:
                return res_min2.x[0]
        elif res_min1.fun < res_min2.fun:
            return res_min1.x[0]
        else:
            return res_min2.x[0]
        
class CostSectionFirst:
    def __init__(self):
        self.coeffs = np.zeros(3, dtype='f8')
        self.current = 0.
    
    def fun(self, theta=None):
        if theta is None:
            theta = self.current
            
        return self.coeffs[0] * np.sin(theta) + \
                self.coeffs[1] * np.cos(theta) + \
                self.coeffs[2]
    
    def grad(self, theta=None):
        if theta is None:
            theta = self.current
            
        return self.coeffs[0] * np.cos(theta) + \
                (-self.coeffs[1]) * np.sin(theta)
    
    def hess(self, theta=None):
        if theta is None:
            theta = self.current
            
        return -1. * (self.fun(theta) - self.coeffs[2])
    
    def minimum(self):
        theta_min = np.arctan2(self.coeffs[0], self.coeffs[1]) + np.pi
        if theta_min < 0.:
            theta_min += 2. * np.pi
        elif theta_min > 2. * np.pi:
            theta_min -= 2. * np.pi

        return theta_min

class CostSectionSecond:
    def __init__(self):
        self.coeffs = np.zeros(3, dtype='f8')
        self.current = 0.
    
    def fun(self, theta=None):
        if theta is None:
            theta = self.current
        
        return self.coeffs[0] * np.sin(2. * theta) + \
                self.coeffs[1] * np.cos(2. * theta) + \
                self.coeffs[2]
    
    def grad(self, theta=None):
        if theta is None:
            theta = self.current
            
        return 2. * self.coeffs[0] * np.cos(2. * theta) + \
                (-2. * self.coeffs[1]) * np.sin(2. * theta)
    
    def hess(self, theta=None):
        if theta is None:
            theta = self.current
            
        return -4. * (self.fun(theta) - self.coeffs[2])
    
    def minimum(self):
        theta_min = np.arctan2(self.coeffs[0], self.coeffs[1]) / 2. + np.pi / 2.
        if theta_min < 0.:
            theta_min += 2. * np.pi
        elif theta_min > 2. * np.pi:
            theta_min -= 2. * np.pi
            
        if abs(theta_min - self.current) > np.pi / 2.:
            if theta_min < np.pi:
                theta_min += np.pi
            else:
                theta_min -= np.pi

        return theta_min
    
class CostSectionSymmetric:
    def __init__(self):
        self.coeffs = np.zeros(3, dtype='f8')
        self.theta_opt = 0.
        self.current = 0.
    
    def fun(self, theta=None):
        if theta is None:
            theta = self.current
        
        x = theta - self.theta_opt
        return self.coeffs[0] * np.cos(2. * x) + \
                self.coeffs[1] * np.cos(x) + \
                self.coeffs[2]
    
    def grad(self, theta=None):
        if theta is None:
            theta = self.current
        
        x = theta - self.theta_opt
        return (-2. * self.coeffs[0]) * np.sin(2. * x) + \
                (-self.coeffs[1]) * np.sin(x)
    
    def hess(self, theta=None):
        if theta is None:
            theta = self.current

        x = theta - self.theta_opt
        return (-4. * self.coeffs[0]) * np.cos(2. * x) + \
                (-self.coeffs[1]) * np.cos(x)
    
    def minimum(self):
        return self.theta_opt


class ManualGeneral(CostSectionGeneral):
    def set_thetas(self, current):
        self.current = current
        self.thetas = np.array([current, current + np.pi / 4., current - np.pi / 4., current + np.pi / 2., current - np.pi / 2.])
        
    def set_coeffs(self, costs):
        z0, z1, z2, z3, z4 = costs

        self.coeffs[4] = (np.sqrt(2) * (z1 + z2) - z4 - z3 - 2 * z0) / (2 * np.sqrt(2) - 4)
        self.coeffs[2] = (
            1
            / 2.0
            * (
                (z3 - z4) * np.cos(self.current)
                + np.sqrt(2) * (z1 + z2) * np.sin(self.current)
                - 2 * np.sqrt(2) * self.coeffs[4] * np.sin(self.current)
            )
        )
        self.coeffs[3] = (
            -1
            / 2.0
            * (
                (z3 - z4) * np.sin(self.current)
                - np.sqrt(2) * (z1 + z2) * np.cos(self.current)
                + 2 * np.sqrt(2) * self.coeffs[4] * np.cos(self.current)
            )
        )
        self.coeffs[1] = (
            np.cos(2 * self.current) * (self.coeffs[2] * np.cos(self.current) - self.coeffs[3] * np.sin(self.current) + self.coeffs[4])
            + np.sin(2 * self.current)
            * (
                (self.coeffs[2] + self.coeffs[3]) * np.cos(self.current) / np.sqrt(2)
                + (self.coeffs[2] - self.coeffs[3]) * np.sin(self.current) / np.sqrt(2)
                + self.coeffs[4]
            )
            - z3 * np.cos(2 * self.current)
            - z1 * np.sin(2 * self.current)
        )
        self.coeffs[0] = (
            np.sin(2 * self.current) * (self.coeffs[2] * np.cos(self.current) - self.coeffs[3] * np.sin(self.current) + self.coeffs[4])
            - np.cos(2 * self.current)
            * (
                (self.coeffs[2] + self.coeffs[3]) * np.cos(self.current) / np.sqrt(2)
                + (self.coeffs[2] - self.coeffs[3]) * np.sin(self.current) / np.sqrt(2)
                + self.coeffs[4]
            )
            - z3 * np.sin(2 * self.current)
            + z1 * np.cos(2 * self.current)
        )
    
class ManualSecond(CostSectionSecond):
    def set_thetas(self, current):
        self.current = current
        self.thetas = np.array([current, current + np.pi / 4., current - np.pi / 4.])
        
    def set_coeffs(self, costs):
        z0, z1, z2 = costs

        self.coeffs[2] = (z1 + z2) / 2.
        
        c = np.cos(2. * self.current)
        s = np.sin(2. * self.current)
        
        self.coeffs[0] = ((z0 - z2) * (c + s) - (z0 - z1) * (c - s)) / 2.
        self.coeffs[1] = ((z0 - z2) * (c - s) + (z0 - z1) * (c + s)) / 2.
    
class ManualFirst(CostSectionFirst):
    def set_thetas(self, current):
        self.current = current
        self.thetas = np.array([current, current + np.pi / 2., current - np.pi / 2.])
        
    def set_coeffs(self, costs):
        z0, z1, z2 = costs

        self.coeffs[2] = (z1 + z2) / 2.
        
        c = np.cos(self.current)
        s = np.sin(self.current)
        
        self.coeffs[0] = ((z0 - z2) * (c + s) - (z0 - z1) * (c - s)) / 2.
        self.coeffs[1] = ((z0 - z2) * (c - s) + (z0 - z1) * (c + s)) / 2.
    
class MatrixInversionMixin:
    def set_coeffs(self, costs):
        self.coeffs = self.inverse_matrix @ costs
    
class InversionGeneral(CostSectionGeneral, MatrixInversionMixin):
    def set_thetas(self, current):
        self.thetas = np.linspace(current - np.pi / 2., current + np.pi / 2., 5, endpoint=True)
        matrix = np.stack((np.sin(2. * self.thetas), np.cos(2. * self.thetas), np.sin(self.thetas), np.cos(self.thetas), np.ones_like(self.thetas)), axis=1)
        self.inverse_matrix = np.linalg.inv(matrix)
        self.current = current
    
class InversionSecond(CostSectionSecond, MatrixInversionMixin):
    def set_thetas(self, current):
        self.thetas = np.linspace(current - np.pi / 4., current + np.pi / 4., 3, endpoint=True)
        matrix = np.stack((np.sin(2. * self.thetas), np.cos(2. * self.thetas), np.ones_like(self.thetas)), axis=1)
        self.inverse_matrix = np.linalg.inv(matrix)
        self.current = current
        
class InversionFirst(CostSectionFirst, MatrixInversionMixin):
    def set_thetas(self, current):
        self.thetas = np.linspace(current - np.pi / 2., current + np.pi / 2., 3, endpoint=True)
        matrix = np.stack((np.sin(self.thetas), np.cos(self.thetas), np.ones_like(self.thetas)), axis=1)
        self.inverse_matrix = np.linalg.inv(matrix)
        self.current = current
        
class FitMixin:
    def __init__(self, points_coeff_ratio=4):
        self.ratio = points_coeff_ratio
    
    def set_coeffs(self, costs):
        # Assuming global cost, cost = 1-prob_dist[0]
        # -> binomial distribution

        def negative_likelihood(coeffs):
            x = self.template_matrix @ coeffs
            return -np.prod(scistats.beta.pdf(x, costs + 1., (1. - costs) + 1.))

        b0 = np.linalg.inv(self.template_matrix[::self.ratio, :]) @ costs[::self.ratio]

        res = sciopt.minimize(negative_likelihood, b0)

        self.coeffs = res.x
    
class FitGeneral(CostSectionGeneral, FitMixin):
    def __init__(self, points_coeff_ratio=4):
        CostSectionGeneral.__init__(self)
        FitMixin.__init__(self, points_coeff_ratio)
        
    def set_thetas(self, current):
        self.thetas = np.linspace(current - np.pi, current + np.pi, 5 * self.ratio, endpoint=False)
        self.template_matrix = np.stack((np.sin(2. * self.thetas), np.cos(2. * self.thetas), np.sin(self.thetas), np.cos(self.thetas), np.ones_like(self.thetas)), axis=1)
        self.current = current
    
class FitSecond(CostSectionSecond, FitMixin):
    def __init__(self, points_coeff_ratio=4):
        CostSectionSecond.__init__(self)
        FitMixin.__init__(self, points_coeff_ratio)
        
    def set_thetas(self, current):
        self.thetas = np.linspace(current - np.pi, current + np.pi, 3 * self.ratio, endpoint=False)
        self.template_matrix = np.stack((np.sin(2. * self.thetas), np.cos(2. * self.thetas), np.ones_like(self.thetas)), axis=1)
        self.current = current
    
class FitFirst(CostSectionFirst, FitMixin):
    def __init__(self, points_coeff_ratio=4):
        CostSectionFirst.__init__(self)
        FitMixin.__init__(self, points_coeff_ratio)
        
    def set_thetas(self, current):
        self.thetas = np.linspace(current - np.pi, current + np.pi, 3 * self.ratio, endpoint=False)
        self.template_matrix = np.stack((np.sin(self.thetas), np.cos(self.thetas), np.ones_like(self.thetas)), axis=1)
        self.current = current
        
class FitSymmetric(CostSectionSymmetric):
    def __init__(self, points_coeff_ratio=4):
        CostSectionSymmetric.__init__(self)
        self.ratio = points_coeff_ratio
        
    def set_thetas(self, current):
        self.thetas = np.linspace(current - np.pi, current + np.pi, 3 * self.ratio, endpoint=False)
        self.theta_opt = current
        self.current = current
    
    def set_coeffs(self, costs):
        # Assuming global cost, cost = 1-prob_dist[0]
        # -> binomial distribution

        def negative_likelihood(params):
            coeffs = params[:3]
            minimum = params[3]
            template_matrix = np.stack((np.cos(2. * (self.thetas - minimum)), np.cos(self.thetas - minimum), np.ones_like(self.thetas)), axis=1)
            x = template_matrix @ coeffs
            return -np.prod(scistats.beta.pdf(x, costs + 1., (1. - costs) + 1.))

        # This class is to be used only when the parameters are close to the optimum
        # -> cost computed at current theta is close to the sum of coeffs
        thetas = self.thetas[::(self.ratio // 2)][:3]
        template_matrix = np.stack((np.cos(2. * (thetas - self.current)), np.cos(thetas - self.current), np.ones_like(thetas)), axis=1)
        coeffs0 = np.linalg.inv(template_matrix) @ costs[::(self.ratio // 2)][:3]
        b0 = np.concatenate((coeffs0, [self.current]))

        res = sciopt.minimize(negative_likelihood, b0)

        self.coeffs = res.x[:3]
        self.theta_opt = res.x[3]
