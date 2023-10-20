import os
import yaml


def get_config(yaml_file=None):
    config_dict = {
        "use_arm": True,
        "opt_arm_pose": False,
        "use_smooth_seq": True,
        "average_cam_sequence": False,
        "img_size": 448,  # 224,
        "focal_length": 2000.0, # 1000.0, # need to be 1000.0 * img_size / 224
        "model_type": "harp",  # ["harp", "html", "nimble"]
        "test_seq": False ,
        "known_appearance": False,
        "load_siren": False,
        "self_shadow": True,
        "pose_already_opt": False,
        "share_light_position": True,
        "eval_mesh": False,
        "use_vert_disp": True,
        "total_epoch": 301,
        # [shape, shape and appearance, appearance only]
        "training_stage": [100, 100, 100],
        "metro_output_dir": "./data/sample_data/1/",
        "image_dir": "./data/sample_data/1/",
        "train_list": ["1", "2"],
        "val_list": ["1", "2"],
        "gt_mesh_dir": "",
        # Output directory
        "base_output_dir": "exp/out_test/",
        "start_from": ""
    }
    if config_dict["use_arm"]:
        # Arm Template
        config_dict["MANO_TEMPLATE"] = "template/arm/arm_template.obj"
        config_dict["uv_mask"] = "template/arm/uv_mask.png"
    else:
        # Hand Template
        config_dict["MANO_TEMPLATE"] = "template/hand/textured_hand.obj"
        config_dict["uv_mask"] = "template/hand/uv_mask.png"

    os.makedirs(config_dict["base_output_dir"], exist_ok=True)
    with open(os.path.join(config_dict["base_output_dir"], "config.yaml"), 'w') as file:
        documents = yaml.dump(config_dict, file)

    return config_dict
