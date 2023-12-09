import torch 
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from barcgp.prediction.covGP.covGPNN_model import COVGPNN
from barcgp.prediction.covGP.covGP_dataGen import SampleGeneartorCOVGP
import secrets

from barcgp.h2h_configs import *
from barcgp.common.utils.file_utils import *
import gpytorch

writer = SummaryWriter(flush_secs=1)


class COVNNLoader:
        def __init__(self,args = None,model_load = False, model_id = 100):            
            self.model = None
            self.train_loader = None
            self.test_loader = None
            self.model_id = model_id                                
            
            if args is None:
                self.train_args = {                    
                "batch_size": 512,
                "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
                "input_dim": 9,
                "n_time_step": 10,
                "latent_dim": 4,
                "gp_output_dim": 3,
                "batch_size": 100                
                }
            else: 
                self.train_args = args      
            
            self.sampGen = None

            # if model_load:
            #     self.model_load()
        def reset_args(self,new_args):
            self.train_args = new_args
      
        def set_train_loader(self,data_loader):
            self.train_loader = data_loader
        def set_test_loader(self,data_loader):
            self.test_loader = data_loader            

        def model_save(self,model_id= None):
            if model_id is None:
                model_id = self.model_id
            # save_dir = os.path.join(model_dir, 'cont_encoder_{model_id}.model')
            save_dir = model_dir+"/"+f"covgpnn_{model_id}.model" 
            torch.save({
            'model_state_dict': self.model.state_dict(),
            'train_args': self.train_args
                }, save_dir)
                
            # torch.save(self.model.state_dict(), save_dir )
            print("model has been saved in "+ save_dir)

        def model_load(self,model_id =None):
            if model_id is None:
                model_id = self.model_id
            saved_data = torch.load(model_dir+"/"+f"covgpnn_{model_id}.model")            
            loaded_args= saved_data['train_args']
            self.reset_args(loaded_args)

            model_state_dict = saved_data['model_state_dict']
            self.model = COVGPNN(self.train_args).to(device='cuda')                
            self.model.to(torch.device("cuda"))
            self.model.load_state_dict(model_state_dict)
            self.model.eval()            

     
        # def train(self,args = None):
        def train(self,sampGen: SampleGeneartorCOVGP):
            train_dataset, val_dataset, test_dataset  = sampGen.get_datasets()
            batch_size = self.train_args["batch_size"]
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            if self.train_loader is None:
                return 
            if args is None:
                args = self.train_args
            
            if self.model is None:     
                model = COVGPNN(args).to(device='cuda')                           
            else:
                model = self.model.to(device='cuda')
            
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=sampGen.output_dim) 
            lr = 0.1            
            optimizer = torch.optim.Adam([{'params': model.covnn.parameters(), 'weight_decay': 1e-4},
                                            {'params': model.gp_layer.hyperparameters(), 'lr': lr * 0.01},
                                            {'params': model.gp_layer.variational_parameters()},
                                            {'params': likelihood.parameters()},
                                        ], lr=lr)
            
            mll = gpytorch.mlls.VariationalELBO(likelihood, model.gp_layer, num_data=len(self.train_loader.dataset))
            ## interation setup
            epochs = tqdm(range(args['max_iter'] // len(self.train_loader) + 1))
            ## training
            count = 0
            for epoch in epochs:
                model.train()
                optimizer.zero_grad()
                train_iterator = tqdm(
                    enumerate(self.train_loader), total=len(self.train_loader), desc="training"
                )

                for i, batch_data in train_iterator:

                    if count > args['max_iter']:
                        print("count exceed")
                        self.model = model
                        return model
                    count += 1

                    train_data = batch_data.to(args['device'])

                    ## reshape
                    batch_size = train_data.size(0)
                    mloss, recon_x, cont_loss, recon_loss = model(train_data)

                    # Backward and optimize
                    optimizer.zero_grad()
                    mloss.mean().backward()
                    optimizer.step()

                    train_iterator.set_postfix({"train_loss": float(mloss.mean())})                    
                writer.add_scalar("train_loss", float(mloss.mean()), epoch)                
                writer.add_scalar("cont_loss", float(cont_loss), epoch)        
                writer.add_scalar("recon_loss", float(recon_loss), epoch)   

                model.eval()
                eval_loss = 0
                cont_eval_loss = 0
                recon_eval_loss = 0
                test_iterator = tqdm(
                    enumerate(self.test_loader), total=len(self.test_loader), desc="testing"
                )

                with torch.no_grad():
                    for i, batch_data in test_iterator:
                        test_data = batch_data.to(args['device'])

                        ## reshape
                        batch_size = test_data.size(0)
                        mloss, recon_x, cont_loss, recon_loss = model(test_data)

                        eval_loss += mloss.mean().item()
                        cont_eval_loss += cont_loss.mean().item()
                        recon_eval_loss += recon_loss.mean().item()
                        test_iterator.set_postfix({"eval_loss": float(mloss.mean())})                        
                        


                eval_loss = eval_loss / len(self.test_loader)
                writer.add_scalar("eval_loss", float(eval_loss), epoch)         
                writer.add_scalar("cont_eval_loss", float(cont_eval_loss), epoch)               
                writer.add_scalar("recon_eval_loss", float(recon_eval_loss), epoch)       
                print("Evaluation Score : [{}]".format(eval_loss))
                
                self.model = model
                
                if epoch%500 == 0:
                    self.model_save(model_id=epoch)

        def get_theta_from_buffer(self,input_for_encoder):      
            if len(input_for_encoder.shape) <3:
                input_for_encoder = input_for_encoder.unsqueeze(dim=0).to(device="cuda")
            else:
                input_for_encoder = input_for_encoder.to(device="cuda")
            theta = self.get_theta(input_for_encoder)
            
            return theta.squeeze()
        
      

        def tsne_evaluate(self):            
            if self.train_loader is None:
                return 
            args = self.train_args
            
            ## training
            count = 0
            train_iterator = tqdm(
                    enumerate(self.train_loader), total=len(self.train_loader), desc="training"
                )
            model = self.model 
            model.eval()
            z_tmp_list = []
            input_list = []
            with torch.no_grad():
                for i, batch_data in train_iterator:    
                                
                    count += 1
                    train_data = batch_data.to(args['device'])                
                    z_tmp = model.get_latent_z(train_data)
                    if z_tmp.shape[0] == args['batch_size']:
                        z_tmp_list.append(z_tmp)
                        input_list.append(train_data)
                stacked_z_tmp = torch.cat(z_tmp_list, dim=0)
                input_list_tmp= torch.cat(input_list, dim=0)

            return stacked_z_tmp, input_list_tmp