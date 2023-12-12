from barcgp.common.utils.file_utils import *
from barcgp.prediction.cont_encoder.cont_encoderTrain import cont_encoder_train
from barcgp.prediction.thetaGP.ThetaGPTrain import thetagp_train
from barcgp.prediction.gp_berkely_train import gp_main
import os

def get_folder_names(directory_path):    
    folder_names = []
    items = os.listdir(directory_path)
    for item in items:
        item_path = os.path.join(directory_path, item)
        if os.path.isdir(item_path):
            folder_names.append(item)
    return folder_names

sub_dirs = get_folder_names(real_dir)


dir_tmp = [os.path.join(real_dir, sub_dirs[0])]
dirs = dir_tmp
for i in range(1,len(sub_dirs)):
    
    dir_tmp = [os.path.join(real_dir, sub_dirs[i])]
    dirs.extend(dir_tmp)



# dirs = [real_dir]
# dirs.extend(timid)

# dirs = timid.copy()

# a_policy_name = 'aggressive_blocking'
# a_policy_dir = os.path.join(train_dir, a_policy_name)
# a_scencurve_dir = os.path.join(a_policy_dir, 'curve')
# a_scenstraight_dir = os.path.join(a_policy_dir, 'straight')
# a_scenchicane_dir = os.path.join(a_policy_dir, 'chicane')

# t_policy_name = 'timid'
# t_policy_dir = os.path.join(train_dir, t_policy_name)
# t_scencurve_dir = os.path.join(t_policy_dir, 'curve')
# t_scenstraight_dir = os.path.join(t_policy_dir, 'straight')
# t_scenchicane_dir = os.path.join(t_policy_dir, 'chicane')

# policy_name = 'race'
# policy_dir = os.path.join(train_dir, policy_name)
# track_scencurve_dir = os.path.join(policy_dir, 'track')


# policy_name = 'wall'
# policy_dir = os.path.join(train_dir, policy_name)
# wall_scencurve_dir = os.path.join(policy_dir, 'curve')
# wall_scenstraight_dir = os.path.join(policy_dir, 'straight')
# wall_scenchicane_dir = os.path.join(policy_dir, 'chicane')

# dirs = [a_scencurve_dir, a_scenstraight_dir, a_scenchicane_dir]
# dirs = [a_scencurve_dir, a_scenstraight_dir, a_scenchicane_dir,t_scencurve_dir, t_scenstraight_dir, t_scenchicane_dir]
# dirs = [track_scencurve_dir]


def main():  
    # print("1~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # print("GP Berkely train init")
    # gp_main(dirs,  realdata = True)
    # print("GP Berkely train Done")
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    
    
    # print("2~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # print("ConstAutoEncoder train init")
    # cont_encoder_train(dirs, realdata=True)
    # print("AutoEncoder train Done")
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    print("3~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Contautoencoder based ThetaGP train init")
    thetagp_train(dirs,realdata = True)
    print("Contautoencoder based ThetaGP train Done")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")




if __name__ == "__main__":
    main()

