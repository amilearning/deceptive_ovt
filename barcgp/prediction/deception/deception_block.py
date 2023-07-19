import torch
from torch import nn 
from torch.nn import functional as F

class DeceptionNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DeceptionNet, self).__init__()        
        # Encoder part of deception net
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, hidden_dim),
            nn.ReLU()
        )        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim)
        )
    
    def forward(self, x):
        # Encode the input
        encoded = self.encoder(x)
        # Decode the encoded representation
        out = self.decoder(encoded)
        return out


class LSTMAutoEncoder(nn.Module):
    def __init__(self,input_size = 8, hidden_size = 5, num_layers = 2):
        super(LSTMAutoEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=False,
        )
        
    def forward(self, x):        
        outputs, (hidden, cell) = self.lstm(x)
        return (hidden, cell)

class LSTMAutoDecoder(nn.Module):
    def __init__(
        self, input_size=8, hidden_size=5, output_size=8, num_layers=2):
        super(LSTMAutoDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden = None):
        if hidden is None:
        # x: tensor of shape (batch_size, seq_length, hidden_size)
            output, (hidden, cell) = self.lstm(x)
        else:
            output, (hidden, cell) = self.lstm(x, hidden)
        prediction = self.fc(output)
        return prediction, (hidden, cell)
    # def forward(self, x):
    #     # x: tensor of shape (batch_size, seq_length, hidden_size)
    #     output, (hidden, cell) = self.lstm(x)

    #     return output, (hidden, cell)
        

class DeceptionBasedModel(nn.Module):
    """LSTM-based Contrasiave Auto Encoder"""

    def __init__(
        self, args):
        """
        args['input_size']: int, batch_size x sequence_length x input_dim
        args['hidden_size']: int, output size of LSTM VAE
        args['latent_size']: int, latent z-layer size
        num_lstm_layer: int, number of layers in LSTM
        """
        super(DeceptionBasedModel, self).__init__()
        self.device = args['device']
        
        
        # dimensions
        self.input_size = args['input_size']
        self.hidden_size = args['hidden_size']
        self.latent_size = args['latent_size']
        self.seq_len = args['seq_len']
        self.num_layers = 2
        self.train_phase = 0
        ## Encoder part 
        self.lstm_enc = LSTMAutoEncoder(
            input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers
        )      
        self.fc21 = nn.Linear(self.hidden_size,self.hidden_size)                        
        self.relu = nn.ReLU()
        self.fc22 = nn.Linear(self.hidden_size, self.latent_size)    

        # deception network part  
        self.deception_encoder = DeceptionNet(input_dim = self.latent_size, hidden_dim = self.hidden_size)        
        
        ### Decoder part 
        self.fc_l2l = nn.Linear(self.latent_size, self.latent_size*self.seq_len)
        self.lstm_dec = LSTMAutoDecoder(
            input_size=self.latent_size, output_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
        )                    
        

    def get_latent_z(self,x):
        if len(x.shape) < 3:
            # evaluation -> batch is 1 
            batch_size = 1 
            seq_len, feature_dim = x.shape
        else:
            batch_size, seq_len, feature_dim = x.shape

        # encode input space to hidden space
        enc_hidden = self.lstm_enc(x)
        enc_h = enc_hidden[0][-1,:,:].view(batch_size, self.hidden_size).to(self.device)
        # extract latent variable z(hidden space to latent space)
        z = self.fc22(self.relu(self.fc21(enc_h)))
        
        return z

    def forward(self, x):
        input = x
        batch_size, seq_len, feature_dim = x.shape                
        if seq_len != self.seq_len:
            print("Warning !! sequence lentgh is not matched")
            return
            
        enc_hidden = self.lstm_enc(x)
        enc_h = enc_hidden[0][-1,:,:].view(batch_size, self.hidden_size).to(self.device)
        # extract latent variable z(hidden space to latent space)
        z = self.fc22(self.relu(self.fc21(enc_h)))
        theta = z
        ##########################################                               
        ##########################################
        if self.train_phase == 1:
            self.phase1_train()            
            z = theta
            # decode latent space to input space
            z = self.fc_l2l(z.to(self.device))        
            z = z.view(batch_size, seq_len, self.latent_size).to(self.device)                
            # reconstruct_output, hidden = self.lstm_dec(z, enc_hidden)
            reconstruct_output, hidden = self.lstm_dec(z)
            # reconstruct_output, hidden = self.lstm_dec(z)            
            losses = self.loss_phase1(reconstruct_output, input)

        elif self.train_phase == 2:
            self.phase2_train()
        ## freeze Encoder network(naive theta) 
        # Activate deception network and unfreeze the others True 
        # maximize the distance between ptb_traj & naive traj and minimize ptb theta norm
            perturb_theta = self.deception_encoder(theta)        
            z = theta
            z = self.fc_l2l(z.to(self.device))        
            z = z.view(batch_size, seq_len, self.latent_size).to(self.device)                            
            reconstruct_output, hidden = self.lstm_dec(z)            

            ptb_z = self.fc_l2l(perturb_theta.to(self.device))        
            ptb_z = ptb_z.view(batch_size, seq_len, self.latent_size).to(self.device)                            
            pertubed_reconstruct_output, hidden = self.lstm_dec(ptb_z)            

            losses = self.loss_phase2(reconstruct_output, pertubed_reconstruct_output, theta, perturb_theta)
        ##########################################
        elif self.train_phase == 3:
            self.phase3_train()
        ## freeze Deception network(naive theta), but activate it  
        # minimize the distance between ptb traj & naive traj
            perturb_theta = self.deception_encoder(theta)        
            z = theta

            z = self.fc_l2l(z.to(self.device))        
            z = z.view(batch_size, seq_len, self.latent_size).to(self.device)                            
            reconstruct_output, hidden = self.lstm_dec(z)            

            ptb_z = self.fc_l2l(perturb_theta.to(self.device))        
            ptb_z = ptb_z.view(batch_size, seq_len, self.latent_size).to(self.device)                            
            pertubed_reconstruct_output, hidden = self.lstm_dec(ptb_z)            

            losses = self.loss_phase3(reconstruct_output, pertubed_reconstruct_output)            

        elif self.train_phase == 4:
            self.phase4_train()            
            z = theta
            z = self.fc_l2l(z.to(self.device))        
            z = z.view(batch_size, seq_len, self.latent_size).to(self.device)                            
            reconstruct_output, hidden = self.lstm_dec(z)                     
            losses = self.loss_phase4(reconstruct_output, input)            

        else: 
            print("Train phase %d is not defined yet",self.train_phase)
              
        m_loss = losses["loss"]
        perturb_loss = losses["perturb_loss"]
        recon_loss = losses["recons_loss"]
        
        return m_loss, perturb_loss, recon_loss
    

    def loss_phase1(self,*args, **kwargs) -> dict:
        """
        Computes loss
        loss = reconstruction loss 
        """
        recons = args[0]
        input = args[1]                        
        recons_loss = F.mse_loss(recons, input)                
        loss = recons_loss 
        return {
            "loss": loss,       
            "perturb_loss" : 0.0,     
            "recons_loss" : recons_loss.detach()                
        }
    
    def loss_phase2(self,*args, **kwargs) -> dict:
        """
        Computes loss
        loss =  reguralization - (reverse)reconstruction loss        
        """
        # x_hat, pertubed_reconstruct_output, theta, perturb_theta
        recons = args[0]
        ptb_recons = args[1]   
        orig_theta = args[2]
        perturb_theta = args[3]

        perturb_weight = 1.0
        perturb_loss = F.mse_loss(perturb_theta,orig_theta)*perturb_weight                
        recons_loss = F.mse_loss(recons, ptb_recons)                
        loss = perturb_loss +1/(recons_loss+1e-20)
        return {
            "loss": loss,            
            "perturb_loss" : perturb_loss.detach(),
            "recons_loss" : recons_loss.detach()                
        }

    def loss_phase3(self,*args, **kwargs) -> dict:
        """
        Computes loss
        loss =  reguralization - (reverse)reconstruction loss        
        """
        # x_hat, pertubed_reconstruct_output, theta, perturb_theta
        recons = args[0]
        ptb_recons = args[1]    
        recons_loss = F.mse_loss(recons, ptb_recons)                
        loss = recons_loss
        return {
            "loss": loss,            
            "perturb_loss" : 0.0,
            "recons_loss" : recons_loss.detach()                
        }

    def loss_phase4(self,*args, **kwargs) -> dict:
        """
        Computes loss
        loss =  reguralization - (reverse)reconstruction loss        
        """
        # x_hat, pertubed_reconstruct_output, theta, perturb_theta
        recons = args[0]
        ptb_recons = args[1]    
        recons_loss = F.mse_loss(recons, ptb_recons)                
        loss = recons_loss
        return {
            "loss": loss,            
            "perturb_loss" : 0.0,
            "recons_loss" : recons_loss.detach()                
        }


    def compute_euclidian_dist(self,A,B):
        # A = torch.randn(512, 5, 9)
        # B = torch.randn(512, 5, 9)
        # Expand dimensions to enable broadcasting
        A_expanded = A.unsqueeze(1)  # [512, 1, 4, 7]
        B_expanded = B.unsqueeze(0)  # [1, 512, 4, 7]

        # Calculate the Euclidean norm between each pair of vectors
        distances = torch.norm(A_expanded - B_expanded, dim=3)  # [512, 512, 4]
        # Sum the Euclidean norms over the sequence dimension
        seq_sum_distances = torch.sum(distances, dim=2)  # [512, 512]
        normalized_tensor = F.normalize(seq_sum_distances, dim=(0, 1))
        
        return normalized_tensor      

    def phase1_train(self):
        ## freeze deception network and unfreeze the others True                 
        self.lstm_enc.requires_grad = True
        self.fc21.requires_grad = True                       
        self.relu.requires_grad = True 
        self.fc22.requires_grad = True
        
        self.deception_encoder.requires_grad = False       

        self.fc_l2l.requires_grad = True
        self.lstm_dec.requires_grad = True
        

    def phase2_train(self):        
        ## freeze Encoder network(naive theta) 
        # Activate deception network and unfreeze the others True 
        # only train the deception module
        self.lstm_enc.requires_grad = False
        self.fc21.requires_grad = False                       
        self.relu.requires_grad = False 
        self.fc22.requires_grad = False

        
        self.deception_encoder.requires_grad =True       

        self.fc_l2l.requires_grad = False
        self.lstm_dec.requires_grad = False 

    def phase3_train(self):
        
        ## freeze Deception network(naive theta), but activate it  
        # Train the decoderpart with perturbed theta 
        self.lstm_enc.requires_grad = False
        self.fc21.requires_grad = False                       
        self.relu.requires_grad = False 
        self.fc22.requires_grad = False

        
        self.deception_encoder.requires_grad =False       

        self.fc_l2l.requires_grad = True
        self.lstm_dec.requires_grad = True   
    
    def phase4_train(self):
        
        ## Fix the decoder and bypass the deception network 
        # Train the encoder part         
        self.lstm_enc.requires_grad = True
        self.fc21.requires_grad = True                       
        self.relu.requires_grad = True 
        self.fc22.requires_grad = True

        self.deception_encoder.requires_grad =False       

        self.fc_l2l.requires_grad = False
        self.lstm_dec.requires_grad = False  
