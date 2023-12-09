from barcgp.common.utils.file_utils import *
from barcgp.prediction.cont_encoder.cont_encoderTrain import cont_encoder_train
from barcgp.prediction.thetaGP.ThetaGPTrain import thetagp_train
from barcgp.prediction.gp_berkely_train import gp_main
from barcgp.prediction.covGP.covGPNN_Train import covGPNN_train
import os

def get_dir(policy_name_, train_dir_):
    policy_dir = os.path.join(train_dir_, policy_name_)
    scencurve_dir = os.path.join(policy_dir, 'curve')
    scenstraight_dir = os.path.join(policy_dir, 'straight')
    scenchicane_dir = os.path.join(policy_dir, 'chicane')
    dirs = [scencurve_dir, scenstraight_dir, scenchicane_dir]
    return dirs



timid_0 = get_dir('timid_0', train_dir)
m500 = get_dir('aggressive_blocking', train_dir)
m100 = get_dir('mild_100', train_dir)
m200 = get_dir('mild_200', train_dir)
m300 = get_dir('mild_300', train_dir)
m1000 = get_dir('mild_1000', train_dir)
m5000 = get_dir('mild_5000', train_dir)
reverse = get_dir('reverse', train_dir)
timid = get_dir('timid', train_dir)
wall_timid  = get_dir('wall_timid',train_dir)
wall_200  = get_dir('wall_200',train_dir)
wall_500  = get_dir('wall_aggressive_blocking',train_dir)
wall_5000  = get_dir('wall_5000',train_dir)
wall_reverse  = get_dir('wall_reverse',train_dir)





# dirs = timid.copy()
# # dirs.extend(m100)
# dirs.extend(m200)
# # dirs.extend(m300)
# dirs.extend(m500)
# # dirs.extend(m1000)
# dirs.extend(m5000)
# dirs.extend(reverse)

# dirs.extend(wall_timid)
# dirs.extend(wall_200)
# dirs.extend(wall_500)
# dirs.extend(wall_5000)
# dirs.extend(wall_reverse)

# dirs = m5000.copy()
dirs = timid_0.copy()
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
    # gp_main(dirs)
    # print("GP Berkely train Done")
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    
    print("2~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("ConstAutoEncoder train init")
    covGPNN_train(dirs)
    print("AutoEncoder train Done")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

  


if __name__ == "__main__":
    main()

