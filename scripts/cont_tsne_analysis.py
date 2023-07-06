from barcgp.common.utils.file_utils import *
from barcgp.prediction.cont_encoder.cont_encoderTrain import cont_encoder_train, tsne_cont_encoder
from barcgp.prediction.thetaGP.ThetaGPTrain import thetagp_train
from barcgp.prediction.gp_berkely_train import gp_main
import os

a_policy_name = 'aggressive_blocking'
a_policy_dir = os.path.join(train_dir, a_policy_name)
a_scencurve_dir = os.path.join(a_policy_dir, 'curve')
a_scenstraight_dir = os.path.join(a_policy_dir, 'straight')
a_scenchicane_dir = os.path.join(a_policy_dir, 'chicane')

t_policy_name = 'timid'
t_policy_dir = os.path.join(train_dir, t_policy_name)
t_scencurve_dir = os.path.join(t_policy_dir, 'curve')
t_scenstraight_dir = os.path.join(t_policy_dir, 'straight')
t_scenchicane_dir = os.path.join(t_policy_dir, 'chicane')

# policy_name = 'race'
# policy_dir = os.path.join(train_dir, policy_name)
# track_scencurve_dir = os.path.join(policy_dir, 'track')


# policy_name = 'wall'
# policy_dir = os.path.join(train_dir, policy_name)
# wall_scencurve_dir = os.path.join(policy_dir, 'curve')
# wall_scenstraight_dir = os.path.join(policy_dir, 'straight')
# wall_scenchicane_dir = os.path.join(policy_dir, 'chicane')

a_dirs = [a_scencurve_dir, a_scenstraight_dir, a_scenchicane_dir]
b_dirs = [t_scencurve_dir, t_scenstraight_dir, t_scenchicane_dir]
# dirs = [a_scencurve_dir, a_scenstraight_dir, a_scenchicane_dir,t_scencurve_dir, t_scenstraight_dir, t_scenchicane_dir]
# dirs = [track_scencurve_dir]


def main():  
    # print("1~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # print("GP Berkely train init")
    # gp_main(dirs)
    # print("GP Berkely train Done")
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    
    print("2~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("ConstAutoEncoder T-SNE init")
    tsne_cont_encoder(a_dirs, b_dirs)
    print("T-SNE Done")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


if __name__ == "__main__":
    main()

