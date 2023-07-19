import torch 
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from barcgp.prediction.deception.deception_block import DeceptionBasedModel
import secrets
import tkinter as tk
import threading
from pynput import keyboard




from barcgp.h2h_configs import *
from barcgp.common.utils.file_utils import *

writer = SummaryWriter(flush_secs=1)

class MyDataset:
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
       
        return x, y



class DeceptionEncoder:
        def __init__(self,args = None,model_load = False, model_id = 100, train_ = True):
            
            
            self.key_interupt = False            
            # Perform any desired actions when the "p" key is pressed
            if train_:
                self.keyboard_thread = threading.Thread(target=lambda: keyboard.Listener(on_press=self.on_key_press).start())
                self.keyboard_thread.start()

            self.train_data = None           
            self.model = None
            self.train_loader = None
            self.test_loader = None
            self.model_id = model_id 
            self.train_phase = 0                               
            
            if args is None:
                self.train_args = {
                "batch_size": 512,
                "device": torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu"),
                "input_size": 9,
                "hidden_size": 8,
                "latent_size": 4,
                "seq_len": 5,
                "learning_rate": 0.0001,
                "max_iter": 60000,
                }
            else: 
                self.train_args = args      
            
            self.input_dim = self.train_args["input_size"]
            self.output_dim = self.train_args["latent_size"]
            self.seq_len = self.train_args["seq_len"]
            
            if model_load:
                self.model_load()
        
        def on_key_press(self,key):   
            if key == keyboard.KeyCode.from_char('p'):
                print("The pause key is pressed!")
                self.key_interupt = True
                # Perform any desired actions when the "p" key is pressed

        def reset_args(self,args):
            self.train_args = args
            self.input_dim = args["input_size"]
            self.output_dim = args["latent_size"]
            self.seq_len = args["seq_len"]

        
        def set_train_loader(self,data_loader):
            self.train_loader = data_loader

        def set_test_loader(self,data_loader):
            self.test_loader = data_loader
            

        def model_save(self,model_id= None):
            if model_id is None:
                model_id = self.model_id
            save_dir = model_dir+f"deceptive_encoder_v{model_id}_p{self.train_phase}.model" 
            torch.save({
            'model_state_dict': self.model.state_dict(),
            'train_args': self.train_args
                }, save_dir)
                
            # torch.save(self.model.state_dict(), save_dir )
            print("model has been saved in "+ save_dir)

        def model_load(self,model_id =None, train_phase = 4):
            if model_id is None:
                model_id = self.model_id
            saved_data = torch.load(model_dir+f"deceptive_encoder_v{model_id}_p{train_phase}.model")            
            loaded_args= saved_data['train_args']
            self.reset_args(loaded_args)

            model_state_dict = saved_data['model_state_dict']
            self.model = DeceptionBasedModel(self.train_args).to(device='cuda')                
            self.model.to(torch.device("cuda"))
            self.model.load_state_dict(model_state_dict)
            self.model.train_phase = 1
            self.model.eval()            

        def get_theta(self,x,np = False):

            z = self.model.get_latent_z(x)
            ###  For TEsting only -> if InputPredictGP is working with the ground truth theta 
            # z = torch.ones(z.shape).to(device="cuda")
            ###
            if torch.is_tensor(z) is False and np is False:                
                z = torch.tensor(z)
            elif torch.is_tensor(z) and np:        
                z = z.cpu().numpy()            
            return z


        def train(self,args = None):
            
            if self.train_loader is None:
                return 
            if args is None:
                args = self.train_args
            
            if self.model is None:     
                self.model = DeceptionBasedModel(args).to(device='cuda')                           
            else:
                self.model = self.model.to(device='cuda')

            optimizer = torch.optim.Adam(self.model.parameters(), lr=args['learning_rate'])

            ## interation setup
            epochs = tqdm(range(args['max_iter'] // len(self.train_loader) + 1))

            user_key_flag = False
            ## training
            count = 0
            ## init training
            self.prev_train_phase = 1
            self.train_phase = 1            
            for epoch in epochs:
                if epoch > 0 and (epoch % 100) ==0:
                    self.model_save(model_id=epoch)
                    self.train_phase = self.train_phase+1
                    if self.train_phase > 4:
                        self.train_phase = 1
                self.model.train_phase = self.train_phase
                self.model.train()
                
                optimizer.zero_grad()
                train_iterator = tqdm(
                    enumerate(self.train_loader), total=len(self.train_loader), desc="training"
                )
                for i, batch_data in train_iterator:
                
                    if count > args['max_iter']:
                        print("count exceed")                            
                        return self.model
                    count += 1

                    train_data = batch_data.to(args['device'])

                    ## reshape
                    batch_size = train_data.size(0)
                    # example_size = past_data.size(1)
                    # image_size = past_data.size(1), past_data.size(2)
                    # past_data = (
                    #     past_data.view(batch_size, example_size, -1).float().to(args['device'])
                    # )
                    # future_data = future_data.view(batch_size, example_size, -1).float().to(args.device)
                    # m_loss, x_hat, mean_consensus_loss
                    mloss, ptb_loss, recon_loss = self.model(train_data)

                    # Backward and optimize
                    optimizer.zero_grad()
                    mloss.mean().backward()
                    optimizer.step()
                    train_iterator.set_postfix({"train_loss": float(mloss.mean())})       

                    if self.key_interupt:
                        
                        # Handle the interruption here
                        # You can change the desired parameters or perform any other action
                        self.model_save(model_id=epoch)
                        print("model has been saved")
                        while True:                        
                            print("Current Training phase number = %d ",self.train_phase)
                            user_input = input("Enter the desired trainging phase number between 1 and 4")                        
                            try:
                                number = int(user_input)
                                if number >= 1 and number <= 4:
                                    print("Received valid number:", number)
                                    # Perform any other actions with the number as needed                                
                                    self.train_phase = number 
                                    self.model.train_phase = self.train_phase
                                    print("Now training phase %d begins", self.train_phase)
                                    self.key_interupt = False
                                    break
                                else:
                                    print("Number out of range. Please enter a number between 1 and 4.")
                            except ValueError:
                                print("Invalid input. Please enter an integer.")
                writer.add_scalar("train_loss", float(mloss.mean()), epoch)                
                writer.add_scalar("ptb_loss", float(ptb_loss), epoch)        
                writer.add_scalar("recon_loss", float(recon_loss), epoch)   
                
                
                ############################################################
                ### Evaluation 
                ############################################################
                self.model.eval()
                eval_loss = 0
                ptb_eval_loss = 0
                recon_eval_loss = 0
                test_iterator = tqdm(
                    enumerate(self.test_loader), total=len(self.test_loader), desc="testing"
                )

                with torch.no_grad():
                    for i, batch_data in test_iterator:
                        test_data = batch_data.to(args['device'])
                        mloss, ptb_loss, recon_loss = self.model(test_data)

                        eval_loss += mloss.mean().item()
                        if isinstance(ptb_loss, float):
                            ptb_eval_loss += ptb_loss
                        else:
                            ptb_eval_loss += ptb_loss.mean().item()
                        recon_eval_loss += recon_loss.mean().item()
                        test_iterator.set_postfix({"eval_loss": float(mloss.mean())})                        
                        
                eval_loss = eval_loss / len(self.test_loader)
                ptb_eval_loss = ptb_eval_loss/len(self.test_loader)
                recon_eval_loss = recon_eval_loss/len(self.test_loader)
                writer.add_scalar("eval_loss", float(eval_loss), epoch)         
                writer.add_scalar("cont_eval_loss", float(ptb_eval_loss), epoch)               
                writer.add_scalar("recon_eval_loss", float(recon_eval_loss), epoch)       
                print("Evaluation Score : [{}]".format(eval_loss))
                self.prev_train_phase = self.model.train_phase
                

                

        
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