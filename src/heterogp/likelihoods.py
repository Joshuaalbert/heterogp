# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import absolute_import

import tensorflow as tf
import numpy as np

from gpflow import settings
from gpflow import logdensities as densities
from gpflow import transforms

from gpflow.decors import params_as_tensors
from gpflow.decors import params_as_tensors_for
from gpflow.decors import autoflow
from gpflow.params import Parameter
from gpflow.params import Parameterized
from gpflow.params import ParamList
from gpflow.quadrature import hermgauss, ndiagquad
from gpflow.likelihoods import Likelihood
from .latent import Latent

class HeteroscedasticLikelihood(Likelihood):
    """
    Represents a generalization of the likelihood to 
    allow input-dependent error, which means each function 
    now accepts more arguments.
    
    The heteroscedastic error should always be the same
    shape as the data vector.
    """
    def __init__(self, log_noise_latent, min_noise=1e-2, name=None):
        super().__init__(name=name)
        self.log_noise_latent = log_noise_latent
        self.min_noise = tf.convert_to_tensor(min_noise,dtype=settings.float_type)
        assert isinstance(log_noise_latent,Latent)
        
    @params_as_tensors
    def hetero_noise(self,X):
        """
        Calculates the heterscedastic variance at points X.
        X must be of shape [S, N, D]
        Returns [S,N,num_latent]
        """
        log_noise,_,_ = self.log_noise_latent.sample_from_conditional(X,full_cov=True)
        hetero_noise = tf.exp(log_noise)
        hetero_noise = tf.where(hetero_noise < self.min_noise, 
                                tf.fill(tf.shape(hetero_noise), self.min_noise), 
                                hetero_noise)
        return hetero_noise
    
    @autoflow((settings.float_type, [None, None]), (tf.int32, []))
    def compute_hetero_noise(self,X, num_samples):
        """
        Computes the hetero_noise at X by sampling `num_sample` times.
        X is shape [N,D]
        """
        X = tf.tile(X[None,:,:],(num_samples,1,1))
        return self.hetero_noise(X) 
        
    @params_as_tensors
    def logp(self, F, Y, hetero_variance=None, **args):
        """The log-likelihood function."""
        raise NotImplemented("sub class must...")

    @params_as_tensors
    def conditional_mean(self, F, hetero_variance=None, **args):  # pylint: disable=R0201
        """The mean of the likelihood conditioned on latent."""
        raise NotImplemented("sub class must...")

    @params_as_tensors
    def conditional_variance(self, F, hetero_variance=None, **args): # pylint: disable=R0201
        """The var of the likelihood conditioned on latent."""
        raise NotImplemented("sub class must...")

    def predict_mean_and_var(self, Fmu, Fvar, hetero_variance=None, **args):
        """
        Given a Normal distribution for the latent function,
        return the mean of Y
        if
            q(f) = N(Fmu, Fvar)
        and this object represents
            p(y|f)
        then this method computes the predictive mean
           \int\int y p(y|f)q(f) df dy
        and the predictive variance
           \int\int y^2 p(y|f)q(f) df dy  - [ \int\int y p(y|f)q(f) df dy ]^2
        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (e.g. Gaussian) will implement specific cases.
        """
        integrand2 = lambda *X, **Ys: self.conditional_variance(*X, **Ys) \
            + tf.square(self.conditional_mean(*X, **Ys))
        E_y, E_y2 = ndiagquad([self.conditional_mean, integrand2],
                self.num_gauss_hermite_points,
                Fmu, Fvar, hetero_variance=hetero_variance, **args)
        V_y = E_y2 - tf.square(E_y)
        return E_y, V_y
    
    def predict_density(self, Fmu, Fvar, Y, hetero_variance=None, **args):
        """
        Given a Normal distribution for the latent function, and a datum Y,
        compute the (log) predictive density of Y.
        i.e. if
            q(f) = N(Fmu, Fvar)
        and this object represents
            p(y|f)
        then this method computes the predictive density
           \int p(y=Y|f)q(f) df
        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (Gaussian, Poisson) will implement specific cases.
        """
        exp_p = ndiagquad(lambda X, Y, **Ys: tf.exp(self.logp(X, Y,**Ys)),
                self.num_gauss_hermite_points,
                Fmu, Fvar, Y=Y, hetero_variance=hetero_variance, **args)
        return tf.log(exp_p)

    @params_as_tensors
    def variational_expectations(self, Fmu, Fvar, Y, hetero_variance=None, **args):
        """
        Compute the expected log density of the data, given a Gaussian
        distribution for the function values.
        if
            q(f) = N(Fmu, Fvar)
        and this object represents
            p(y|f)
        then this method computes
           \int (\log p(y|f)) q(f) df.
        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (Gaussian, Poisson) will implement specific cases.
        """
        return ndiagquad(self.logp,
                self.num_gauss_hermite_points,
                Fmu, Fvar, Y=Y, hetero_variance=hetero_variance, **args)
        
class HeteroscedasticGaussian(HeteroscedasticLikelihood):
    def __init__(self, log_noise_latent, name=None):
        super().__init__(log_noise_latent, name=name)
        
    @params_as_tensors
    def logp(self, F, Y, hetero_variance=None,**unused_args):
        """The log-likelihood function."""
        return densities.gaussian(F, Y, hetero_variance)

    @params_as_tensors
    def conditional_mean(self, F, unused_hetero_variance=None, **unused_args):  # pylint: disable=R0201
        """The mean of the likelihood conditioned on latent."""
        return F

    @params_as_tensors
    def conditional_variance(self, F, hetero_variance=None, **unused_args):
        return hetero_variance
