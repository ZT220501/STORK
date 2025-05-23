# We provide four scripts for evaluating metrics:
fid_clipscore_launch=scripts/bash_run_inference_metric.sh
geneval_launch=scripts/bash_run_inference_metric_geneval.sh
dpg_launch=scripts/bash_run_inference_metric_dpg.sh
image_reward_launch=scripts/bash_run_inference_metric_imagereward.sh

# Use following format to metric your models:
# bash $correspoinding_metric_launch $your_config_file_path $your_relative_pth_file_path


# select method from [flow_dpm-solver, flow_euler, flow_rock4-2nd-2, flow_rkg-2nd]
# By default, INTRA_S=100. We set it to 5 for best generation
# cfg_scale=4.5  #default
# sample_nums=30000  #default
export INTRA_S=5
export SEED=0
np=8 bs=20 sampling_algo="flow_rock4-2nd-2" step=19 bash $fid_clipscore_launch \
    configs/sana_config/512ms/Sana_600M_img512.yaml \
    output/Sana_600M_512px/checkpoints/Sana_600M_512px_MultiLing.ptsh
