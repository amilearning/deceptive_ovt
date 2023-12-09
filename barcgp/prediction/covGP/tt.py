import math
# import tqdm
from tqdm import tqdm
import torch
import gpytorch
from matplotlib import pyplot as plt
import torch.nn as nn
# Make plots inline
# %matplotlib inline

import urllib.request
import os
from scipy.io import loadmat
from math import floor


# this is for running the notebook in our testing framework
smoke_test = True # ('CI' in os.environ)


if not smoke_test and not os.path.isfile('../elevators.mat'):
    print('Downloading \'elevators\' UCI dataset...')
    urllib.request.urlretrieve('https://drive.google.com/uc?export=download&id=1jhWL3YUHvXIaftia4qeAyDwVxo6j1alk', '../elevators.mat')


if smoke_test:  # this is for running the notebook in our testing framework
    X, y = torch.randn(2000, 3), torch.randn(2000)
else:
    data = torch.Tensor(loadmat('../elevators.mat')['data'])
    X = data[:, :-1]
    X = X - X.min(0)[0]
    X = 2 * (X / X.max(0)[0]) - 1
    y = data[:, -1]


train_n = int(floor(0.8 * len(X)))
train_x = X[:train_n, :].contiguous()
train_y = y[:train_n].contiguous()

test_x = X[train_n:, :].contiguous()
test_y = y[train_n:].contiguous()

# if torch.cuda.is_available():
#     train_x, train_y, test_x, test_y = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda()

data_dim = train_x.size(-1)

class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(data_dim, 1000))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(1000, 500))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('linear3', torch.nn.Linear(500, 50))
        self.add_module('relu3', torch.nn.ReLU())
        self.add_module('linear4', torch.nn.Linear(50, 2))

feature_extractor = LargeFeatureExtractor()

class GPRegressionModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.GridInterpolationKernel(
                gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2)),
                num_dims=2, grid_size=100
            )

            # self.aux_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=3)
            # self.aux_kernel.lengthscale = 1.0
            self.aux_kernel = gpytorch.kernels.LinearKernel(num_dimensions = 3)
            self.aux_kernel.variance = 1.0
            
            
            # b.base_kernel.outputscale = self.covar_module.base_kernel.outputscale.detach().clone()
            # b.base_kernel.base_kernel.lengthscale = self.covar_module.base_kernel.base_kernel.lengthscale.detach().clone()
            self.feature_extractor = feature_extractor

            # This module will scale the NN features so that they're nice values
            self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)
            
        
        def compute_cov(x): 
            
            return 1

        def forward(self, x, train= False):
            # We're first putting our data through a deep net (feature extractor)
            projected_x = self.feature_extractor(x)
            projected_x = self.scale_to_bounds(projected_x)  # Make the NN values "nice"
            
            # b = gpytorch.kernels.GridInterpolationKernel(
            #     gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=3)),
            #     num_dims=3, grid_size=100
            # )
            # b.base_kernel.outputscale[:] = self.covar_module.base_kernel.outputscale.detach().clone()
            # b.base_kernel.base_kernel.lengthscale[:] = self.covar_module.base_kernel.base_kernel.lengthscale.detach().clone()
            # cov_input = b(x)
            
            mean_x = self.mean_module(projected_x)
            covar_x = self.covar_module(projected_x)
            
            if train:    
                # input_cov = self.aux_kernel(x)
                
                output_cov = self.covar_module(projected_x)
                

                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x), torch.corrcoef(x), output_cov.evaluate()             
            else:        
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = GPRegressionModel(train_x, train_y, likelihood)

# if torch.cuda.is_available():
#     model = model.cuda()
#     likelihood = likelihood.cuda()

training_iterations = 10 if smoke_test else 60

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam([
    {'params': model.feature_extractor.parameters()},
    {'params': model.covar_module.parameters()},
    {'params': model.mean_module.parameters()},
    {'params': model.likelihood.parameters()},
], lr=0.01)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
mseloss = nn.MSELoss()
cov_loss_weight = 0.5
def train():
    iterator = tqdm(range(training_iterations))
    for i in iterator:
        # Zero backprop gradients
        optimizer.zero_grad()
        # Get output from model
        output, input_cov, covar_x = model(train_x, train=True)
        # Calc loss and backprop derivatives
        cov_loss = mseloss(input_cov, covar_x)*cov_loss_weight
        loss = -mll(output, train_y) + cov_loss        
        print("total loss = {:.5f}, total cov_loss ={:.5f}".format( loss.item(), cov_loss.item()))        
        
        loss.backward()
        iterator.set_postfix(loss=loss.item())
        optimizer.step()

train()


model.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
    preds= model(test_x)
print('Test MAE: {}'.format(torch.mean(torch.abs(preds.mean - test_y))))