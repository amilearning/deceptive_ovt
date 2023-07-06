#!/usr/bin/env python3

from barcgp.common.utils.file_utils import *
import numpy as np
import torch
from barcgp.prediction.cont_encoder.cont_encoderdataGen import SampleGeneartorContEncoder
from barcgp.prediction.cont_encoder.cont_policyEncoder import ContPolicyEncoder
from torch.utils.data import DataLoader, random_split


# Training
def cont_encoder_train(dirs):

    sampGen = SampleGeneartorContEncoder(dirs, randomize=True)
    
    sampGen.plotStatistics()
    
    if not dir_exists(dirs[0]):
        raise RuntimeError(
            f"Directory: {dirs[0]} does not exist, need to train using `gen_training_data` first")

    train_dataset, val_dataset, test_dataset  = sampGen.get_datasets()
    args_ =  {
                "batch_size": 512,
                "device": torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu"),
                "input_size": 9,
                "hidden_size": 8,
                "latent_size": 4,
                "learning_rate": 0.0005,
                "max_iter": 180000,
                "seq_len" :5
            }
    batch_size = args_["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



    
    policy_encoder = ContPolicyEncoder(args= args_)    
    policy_encoder.set_train_loader(train_loader)
    policy_encoder.set_test_loader(test_loader)

    policy_encoder.train(args= args_)
    
    create_dir(path=model_dir)
    policy_encoder.model_save()
    


# T-SNE analysis 

def tsne_cont_encoder(a_dirs, b_dirs):
    a_sampGen = SampleGeneartorContEncoder(a_dirs, randomize=True)
    b_sampGen = SampleGeneartorContEncoder(b_dirs, randomize=True)
    
    a_train_dataset, a_val_dataset, a_test_dataset  = a_sampGen.get_datasets(filter= True)
    b_train_dataset, b_val_dataset, b_test_dataset  = b_sampGen.get_datasets(filter= True)

    args_ =  {
            "batch_size": 512,
            "device": torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu"),
            "input_size": 9,
            "hidden_size": 8,
            "latent_size": 4,
            "learning_rate": 0.0001,
            "max_iter": 240000,
            "seq_len" :5
        }
    batch_size = args_["batch_size"]
    
    a_train_loader = DataLoader(a_train_dataset, batch_size=batch_size, shuffle=True)    
    a_test_loader = DataLoader(a_test_dataset, batch_size=batch_size, shuffle=False)
    
    b_train_loader = DataLoader(b_train_dataset, batch_size=batch_size, shuffle=True)    
    b_test_loader = DataLoader(b_test_dataset, batch_size=batch_size, shuffle=False)

    a_policy_encoder = ContPolicyEncoder(args= args_)    
    a_policy_encoder.set_train_loader(a_train_loader)
    a_policy_encoder.set_test_loader(a_test_loader)
    a_policy_encoder.model_load()

    b_policy_encoder = ContPolicyEncoder(args= args_)    
    b_policy_encoder.set_train_loader(b_train_loader)
    b_policy_encoder.set_test_loader(b_test_loader)
    b_policy_encoder.model_load()

    print("a stacked z init")
    a_stacked_z = a_policy_encoder.tsne_evaluate()
    print("b stacked z init")
    b_stacked_z = b_policy_encoder.tsne_evaluate()

    stacked_z = torch.vstack([a_stacked_z,b_stacked_z]).cpu()    
    # label generate 
    a_y_label = torch.ones(a_stacked_z.shape[0])
    b_y_label = torch.ones(b_stacked_z.shape[0])*2.0
    y_label = torch.hstack([a_y_label,b_y_label]).cpu().numpy()
    ###################################
    ###################################
    ########## TSNE_analysis ##########
    ###################################
    ###################################
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    for i in range(1):
        ###################################        
        dim = 2        
        perplexity_ = 150
        n_iter_ = 800        

        ###################################
        tsne_model = TSNE(n_components=dim,perplexity=perplexity_, verbose= 2,n_iter=n_iter_)        
        print("t-SNE optimization begin")
        theta_2d = tsne_model.fit_transform(stacked_z)
        print("t-SNE optimization done")
        
        if dim >2:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(theta_2d[:, 0], theta_2d[:, 1],theta_2d[:, 2] ,c=y_label, cmap='viridis')
        else:
            fig, ax = plt.subplots()
            ax.scatter(theta_2d[:, 0], theta_2d[:, 1], c=y_label, cmap='viridis')            
            plt.show()
            # cbar = plt.colorbar()
            # cbar.set_label('Color Bar Label')
        
    


# if __name__ == "__main__":
#     main()
