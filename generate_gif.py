import os
import imageio

def create_gif(folder_path, fps=10):
    """
    Create a GIF from a sequence of images in a folder.

    Parameters:
    folder_path (str): Path to the folder containing image files.
    folder_path (str): Path to save the output GIF file.
    fps (int): Frames per second for the GIF.
    """

    images = []
    images_1=[]
    file_paths=[]
    # Assuming files are named in a sorted manner that represents the sequence
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.endswith('.png'):
            if 'rgb_all_val.png' in file_name:
                file_path = os.path.join(folder_path, file_name)
                file_paths.append(file_path)

    # file_name 的例子 '1_rgb_all_val.png', '12_rgb_all_val.png', 需要自定义排序方法
    file_paths.sort(key=lambda x:int(x.split('/')[-1].split('_')[0]))
    for idx,file_path in enumerate(file_paths):
        # images.append(imageio.imread(file_path))
        img = imageio.imread(file_path)[:,:,:3]
        if idx<150:images.append(img)
        else:images_1.append(img)

    # Convert images to GIF
    imageio.mimsave(os.path.join(folder_path,'output_train.gif'), images, fps=fps)
    imageio.mimsave(os.path.join(folder_path,'output_val.gif'), images_1, fps=fps)

    images = []
    images_1=[]
    file_paths=[]
    # Assuming files are named in a sorted manner that represents the sequence
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.endswith('.png'):
            if 'y_true_val.png' in file_name:
                file_path = os.path.join(folder_path, file_name)
                file_paths.append(file_path)

    # file_name 的例子 '1_rgb_all_val.png', '12_rgb_all_val.png', 需要自定义排序方法
    file_paths.sort(key=lambda x:int(x.split('/')[-1].split('_')[0]))
    for idx,file_path in enumerate(file_paths):
        # images.append(imageio.imread(file_path))
        img = imageio.imread(file_path)[:,:,:3]
        if idx<150:images.append(img)
        else:images_1.append(img)

    # Convert images to GIF
    imageio.mimsave(os.path.join(folder_path,'output_gt_train.gif'), images, fps=fps)
    imageio.mimsave(os.path.join(folder_path,'output_gt_val.gif'), images_1, fps=fps)

# Example usage
create_gif("/home1/jo_891/data1/harp/exp/out_test2/val_output", fps=30)
