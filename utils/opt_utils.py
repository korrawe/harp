import torch



########### Helper Functions ##################

def get_mano_vert_colors(mano_layer):
    device = "cuda"
    mano_layer = mano_layer.to(device)
    batch_size = 1
    # Model para initialization:
    shape = torch.zeros(batch_size, 10).to(device)
    rot = torch.zeros(batch_size, 3).to(device)
    pose = torch.zeros(batch_size, 45).to(device)
    trans = torch.zeros(batch_size, 3).to(device)

    mano_vertices, _ = mano_layer(torch.cat((rot, pose), 1), shape, trans)
    mano_vertices = mano_vertices[0].detach().cpu().numpy()
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(mano_vertices)
    vertex_colors = scaler.transform(mano_vertices)
    return vertex_colors

def get_upscale_mano_vert_colors(upscale_vertices):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(upscale_vertices)
    vertex_colors = scaler.transform(upscale_vertices)
    return vertex_colors

class PyTMinMaxScaler(object):
    """
    Transforms each channel to the range [0, 1].
    """
    def __call__(self, tensor):
        dist = (tensor.max(dim=1, keepdim=True)[0] - tensor.min(dim=1, keepdim=True)[0])
        dist[dist==0.] = 1.
        scale = 1.0 /  dist
        tensor.mul_(scale).sub_(tensor.min(dim=1, keepdim=True)[0])
        return tensor

def scale_value(tensor):
    scaler = PyTMinMaxScaler()
    return scaler(tensor)

########### End Helper Functions ##################




def test_render(sample_sequence, mano_layer, phong_renderer, img_width, img_height, device='cuda'):
    # Copy from the main function without tested
    mano_faces=mano_layer.th_faces.detach().cpu()
    # Show mask
    mask = sample_sequence['mask'][0]

    sample_mano = sample_sequence['mano'][0]
    # import pdb; pdb.set_trace()
    cam = sample_mano['cam']
    res = 224  # img.shape[1]
    focal_length = 1000
    camera_t = np.array([cam[1], cam[2], 2*focal_length/(res * cam[0] +1e-9)])
    camera_t = torch.Tensor(camera_t).unsqueeze(0)

    hand_verts, hand_joints = mano_layer(torch.cat((torch.tensor(sample_mano['rot']), torch.tensor(sample_mano['pose'])),1),
        torch.tensor(sample_mano['shape']),
        torch.tensor(sample_mano['trans']))

    hand_verts = hand_verts /1000.0

    # import pdb; pdb.set_trace()
    # hand_verts[:, :, 0] = -hand_verts[:, :, 0]
    # hand_verts[:, :, 1] = -hand_verts[:, :, 1]

    test_rgb = renderer_helper.render_rgb_test(phong_renderer, hand_verts, mano_faces, camera_t, img_width, img_height, device)
    
    import pdb; pdb.set_trace()
    fig = plt.figure(figsize=(10, 5))
    # plt.title("iter %d RGB loss: %f" % (i, rgb_loss))
    ax1 = fig.add_subplot(121)
    # ax1.axis('off')
    idx = 0
    ax1.imshow(test_rgb[idx].detach().cpu().numpy().clip(0,1))

    ax2 = fig.add_subplot(122)
    ax2.imshow(mask)
    plt.show()