import torch
import os
import pickle
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def load_img(img_path, torch_tensor=False, downsample_factor=1, load_mask=False, erode=False):
    
    if load_mask:
        img = np.asarray(Image.open(img_path).convert('L')) / 255
        img = img[::downsample_factor, ::downsample_factor, None]

        # Erode mask
        if erode:
            kernel = np.ones((3,3), np.uint8)
            img = cv2.erode(img, kernel, iterations=2)

    else:
        img = np.asarray(Image.open(img_path).convert('RGB')) / 255
        img = img[::downsample_factor, ::downsample_factor, 0:3]

    # Images should be of size [224, 224, 3]
    if torch_tensor:
        img = torch.Tensor(img)  # .unsqueeze(0) #.transpose(1, 3).transpose(2, 3)
          
    return img

class ImagesDataset(Dataset):
    '''Pytorch Dataset to load ground truth images efficiently
    '''
    def __init__(self, images_paths, mask_paths, downsample_factor):
        
        self.image_paths = images_paths
        self.mask_paths = mask_paths
        self.downsample_factor = downsample_factor
        # self.samples = frame_ids
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, ix):
        fid = ix
        col_img = load_img(self.image_paths[fid], downsample_factor=self.downsample_factor, torch_tensor=True)
        mask_img = load_img(self.mask_paths[fid], downsample_factor=self.downsample_factor, torch_tensor=True, load_mask=True)
        mask_img_eroded = load_img(self.mask_paths[fid], downsample_factor=self.downsample_factor, torch_tensor=True, load_mask=True, erode=True)
        # mask_img_eroded = load_img(self.mask_paths[fid], downsample_factor=2, torch_tensor=True, load_mask=True, erode=True)
        return fid, col_img, mask_img, mask_img_eroded


def combine_dict_to_batch(mano_dict):
    out = {}
    keys = mano_dict[0].keys()
    for k in keys:
        out[k] = []
    dict_size = len(mano_dict)
    for idx in range(dict_size):
        # print("idx", idx)
        for k in keys:
            if k == 'cam':
                out[k].append(torch.from_numpy(mano_dict[idx][k]))
            elif k == 'seq':
                out[k].append(mano_dict[idx][k])
            else:
                out[k].append(torch.from_numpy(mano_dict[idx][k].squeeze(0)))

    for k in keys:
        if k != 'seq':
            out[k] = torch.stack(out[k])
    return out


def load_multiple_sequences(metro_output_dir, image_dir, max_size=0, val=False, val_size=0,
        average_cam_sequence=False,
        train_list = ["1", "2", "3", "4", "5"],
        val_list = ["6", "7", "8", "9"],
        use_smooth_seq=False,
        model_type='harp'):
    '''
        max_size: maximum training size. If it is 0, take every files in the directory
        val: If val is True, then split 10% of the data as validation. If max_size is set, take 10% of the max_size 
            immediately after the training set for validation.

    '''
    if use_smooth_seq:
        pkl_folder = "metro_mano_smooth"
    else:
        pkl_folder = "metro_mano"
    
    if model_type == 'nimble':
        if use_smooth_seq:
            pkl_folder = "nimble_metro_mano_smooth"
        else:
            pkl_folder = "nimble_metro_mano"

    data = {'img': [], 'mask': [], 'mano': []}
    DOWNSAMPLE_FACTOR = 1
    ONE_FRAME = False
    DATA_LIMIT = 500
    # mano_param keys: 'joints', 'verts', 'rot', 'pose', 'shape', 'trans'

    image_name_list = []
    for seq in train_list:
        for filename in os.listdir(os.path.join(metro_output_dir, seq, pkl_folder)):
            # example: ../data/sequence/final/1/1/metro_mano/0001_mano.pkl
            if filename.endswith(".pkl"):
                image_name_list.append((seq, filename[:-9]))
    image_name_list.sort()

    val_image_name_list = []
    if len(val_list) > 0:
        for seq in val_list:
            for filename in os.listdir(os.path.join(metro_output_dir, seq, pkl_folder)):
                # example: /../data/sequence/final/1/1/metro_mano/0001_mano.pkl
                if filename.endswith(".pkl"):
                    val_image_name_list.append((seq, filename[:-9])) # 0001
        val_image_name_list.sort()

    image_paths = []
    mask_paths = []
    val_image_paths = []
    val_mask_paths = []
    mano_list = []
    val_mano_list = []
    cam_list = {}
    for idx, (seq, image_name) in enumerate(image_name_list):
        if ONE_FRAME and idx >= 1:
            continue
        # if not (500 <= idx and idx < 1000):
        #     continue
        img_filename = os.path.join(image_dir, seq, "unscreen_cropped", image_name + ".jpg")
        mask_filename = os.path.join(image_dir, seq, "mask", image_name + "_mask.jpg")
        mano_filename = os.path.join(metro_output_dir, seq, pkl_folder, image_name + "_mano.pkl")
        # MANO
        with open(mano_filename, 'rb') as f:
            mano_param = pickle.load(f)
            mano_param["seq"] = seq
            if not seq in cam_list:
                cam_list[seq] = []
            cam_list[seq].append(mano_param["cam"])
                
        image_paths.append(img_filename)
        mask_paths.append(mask_filename)
        mano_list.append(mano_param)

    # Val
    for idx, (seq, image_name) in enumerate(val_image_name_list):
        if ONE_FRAME and idx >= 1:
            continue
        # if not (500 <= idx and idx < 1000):
        #     print("skip", idx)
        #     continue
        img_filename = os.path.join(image_dir, seq, "unscreen_cropped", image_name + ".jpg")
        mask_filename = os.path.join(image_dir, seq, "mask", image_name + "_mask.jpg")
        mano_filename = os.path.join(metro_output_dir, seq, pkl_folder, image_name + "_mano.pkl")
        # MANO
        with open(mano_filename, 'rb') as f:
            mano_param = pickle.load(f)
            mano_param["seq"] = seq
            if not seq in cam_list:
                cam_list[seq] = []
            cam_list[seq].append(mano_param["cam"])
        val_image_paths.append(img_filename)
        val_mask_paths.append(mask_filename)
        val_mano_list.append(mano_param)

    # Force the same camera for the entire sequence
    if average_cam_sequence:
        average_cam_dict = {}
        # Compute average camera position
        for seq_name, seq_cam_lists in cam_list.items():
            average_cam = np.mean(seq_cam_lists, axis=0)
            average_cam_dict[seq_name] = average_cam
        # Re-assign the average cam to param["cam"]
        for mano_param in mano_list:
            mano_param["cam"] = average_cam_dict[mano_param["seq"]]
        # Re-assign the average cam to param["cam"] in val
        for mano_param in val_mano_list:
            mano_param["cam"] = average_cam_dict[mano_param["seq"]]

    if len(val_image_name_list) == 0:
        # Let val be the training set
        val_image_paths = image_paths
        val_mask_paths = mask_paths
        val_mano_list = mano_list
        
    mano_params = combine_dict_to_batch(mano_list)
    val_mano_params = combine_dict_to_batch(val_mano_list)
    images_dataset = ImagesDataset(image_paths, mask_paths, downsample_factor=DOWNSAMPLE_FACTOR)
    val_images_dataset = ImagesDataset(val_image_paths, val_mask_paths, downsample_factor=DOWNSAMPLE_FACTOR)

    return mano_params, images_dataset, val_mano_params, val_images_dataset


def load_sample_sequence(metro_output_dir, image_dir, max_size=0, val=False, val_size=0, average_cam_sequence=False):
    '''
        max_size: maximum training size. If it is 0, take every files in the directory
        val: If val is True, then split 10% of the data as validation. If max_size is set, take 10% of the max_size 
            immediately after the training set for validation.

    '''
    data = {'img': [], 'mask': [], 'mano': []}
    DOWNSAMPLE_FACTOR = 1
    # mano_param keys: 'joints', 'verts', 'rot', 'pose', 'shape', 'trans'
    # for filename in os.listdir(args.image_file_or_path):
    
    image_name_list = []
    for filename in os.listdir(metro_output_dir):
        # if filename.endswith(".png") or filename.endswith(".jpg") and 'mask' not in filename:
        if filename.endswith(".pkl"):
            image_name_list.append(filename[:-9])
    image_name_list.sort()

    if val:
        if max_size == 0:
            # If max_size = 0, ignore val_size
            # 90% for training, 10% for val
            max_size = int((len(image_name_list) * 9) // 10)
            val_size = int(len(image_name_list) // 10)
        elif val_size == 0:
            # 100% max_size for training, 10% for val (total 100%), assuming total size is very large
            val_size = len(image_name_list) - max_size

    if max_size == 0:
        max_size = len(image_name_list)

    image_paths = []
    mask_paths = []
    val_image_paths = []
    val_mask_paths = []
    mano_list = []
    val_mano_list = []
    cam_list = {}
    for idx, image_name in enumerate(image_name_list):
        if "sequence" in metro_output_dir or "interhand" in metro_output_dir : # for captured videos
            img_filename = image_dir + image_name + ".jpg"
            mask_filename = image_dir + image_name + "_mask.jpg"
        else:
            img_filename = image_dir + image_name + "_cropped.jpg"
            mask_filename = image_dir + image_name + "_mask.jpg"
        mano_filename = metro_output_dir + image_name + "_mano.pkl"

        # MANO
        with open(mano_filename, 'rb') as f:
            mano_param = pickle.load(f)
            seq = "0"
            mano_param["seq"] = seq
            if not seq in cam_list:
                cam_list[seq] = []
            cam_list[seq].append(mano_param["cam"])

        if len(image_paths) < max_size:
            image_paths.append(img_filename)
            mask_paths.append(mask_filename)
            mano_list.append(mano_param)
        elif len(image_paths) >= max_size and len(val_image_paths) < val_size:
            val_image_paths.append(img_filename)
            val_mask_paths.append(mask_filename)
            val_mano_list.append(mano_param)
        else:
            break

    # Force the same camera for the entire sequence
    if average_cam_sequence:
        average_cam_dict = {}
        # Compute average camera position
        for seq_name, seq_cam_lists in cam_list.items():
            average_cam = np.mean(seq_cam_lists, axis=0)
            average_cam_dict[seq_name] = average_cam
        # Re-assign the average cam to param["cam"]
        for mano_param in mano_list:
            mano_param["cam"] = average_cam_dict[mano_param["seq"]]
        # Re-assign the average cam to param["cam"] in val
        for mano_param in val_mano_list:
            mano_param["cam"] = average_cam_dict[mano_param["seq"]]

    if val_size == 0:
        # This can only happen if total data is < 10. Only used for debugging.
        # Let val be the training set
        val_image_paths = image_paths
        val_mask_paths = mask_paths
        val_mano_list = mano_list
        
    mano_params = combine_dict_to_batch(mano_list)
    val_mano_params = combine_dict_to_batch(val_mano_list)
    images_dataset = ImagesDataset(image_paths, mask_paths, downsample_factor=DOWNSAMPLE_FACTOR)
    val_images_dataset = ImagesDataset(val_image_paths, val_mask_paths, downsample_factor=DOWNSAMPLE_FACTOR)
    if val:
        return mano_params, images_dataset, val_mano_params, val_images_dataset
    return mano_params, images_dataset

def batch_to_tensor_device(batch, device):
    def to_tensor(arr):
        if isinstance(arr, int):
            return arr
        if isinstance(arr, torch.Tensor):
            return arr.to(device)
        if arr.dtype == np.int64:
            arr = torch.from_numpy(arr)
        else:
            arr = torch.from_numpy(arr).float()
        return arr

    for key in batch:
        if isinstance(batch[key], np.ndarray):
            batch[key] = to_tensor(batch[key]).to(device)
        elif isinstance(batch[key], list):
            for i in range(len(batch[key])):
                if isinstance(batch[key][i], list):
                    for j in range(len(batch[key][i])):
                        if isinstance(batch[key][i][j], np.ndarray):
                            batch[key][i][j] = to_tensor(batch[key][i][j]).to(device)
                else:
                    batch[key][i] = to_tensor(batch[key][i]).to(device)
        elif isinstance(batch[key], dict):
            batch[key] = batch_to_tensor_device(batch[key], device)
        elif isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
    
    return batch