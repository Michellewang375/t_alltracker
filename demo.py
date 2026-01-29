#Alltracker code

import torch
import cv2
import argparse
import utils.saveload
import utils.basic
import utils.improc
import PIL.Image
import numpy as np
import os
from prettytable import PrettyTable
import time


def read_mp4(name_path):
    vidcap = cv2.VideoCapture(name_path)
    framerate = int(round(vidcap.get(cv2.CAP_PROP_FPS)))
    print('framerate', framerate)
    frames = []
    while vidcap.isOpened():
        ret, frame = vidcap.read()
        if ret == False:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    vidcap.release()
    return frames, framerate

def draw_pts_gpu(rgbs, trajs, visibs, colormap, rate=1, bkg_opacity=0.5):
    device = rgbs.device
    T, C, H, W = rgbs.shape
    trajs = trajs.permute(1,0,2) # N,T,2
    visibs = visibs.permute(1,0) # N,T
    N = trajs.shape[0]
    colors = torch.tensor(colormap, dtype=torch.float32, device=device)  # [N,3]

    rgbs = rgbs * bkg_opacity # darken, to see the point tracks better
    
    opacity = 1.0
    if rate==1:
        radius = 1
        opacity = 0.9
    elif rate==2:
        radius = 1
    elif rate== 4:
        radius = 2
    elif rate== 8:
        radius = 4
    else:
        radius = 6
    sharpness = 0.15 + 0.05 * np.log2(rate)
    
    D = radius * 2 + 1
    y = torch.arange(D, device=device).float()[:, None] - radius
    x = torch.arange(D, device=device).float()[None, :] - radius
    dist2 = x**2 + y**2
    icon = torch.clamp(1 - (dist2 - (radius**2) / 2.0) / (radius * 2 * sharpness), 0, 1)  # [D,D]
    icon = icon.view(1, D, D)
    dx = torch.arange(-radius, radius + 1, device=device)
    dy = torch.arange(-radius, radius + 1, device=device)
    disp_y, disp_x = torch.meshgrid(dy, dx, indexing="ij")  # [D,D]
    for t in range(T):
        mask = visibs[:, t]  # [N]
        if mask.sum() == 0:
            continue
        xy = trajs[mask, t] + 0.5  # [N,2]
        xy[:, 0] = xy[:, 0].clamp(0, W - 1)
        xy[:, 1] = xy[:, 1].clamp(0, H - 1)
        colors_now = colors[mask]  # [N,3]
        N = xy.shape[0]
        cx = xy[:, 0].long()  # [N]
        cy = xy[:, 1].long()
        x_grid = cx[:, None, None] + disp_x  # [N,D,D]
        y_grid = cy[:, None, None] + disp_y  # [N,D,D]
        valid = (x_grid >= 0) & (x_grid < W) & (y_grid >= 0) & (y_grid < H)
        x_valid = x_grid[valid]  # [K]
        y_valid = y_grid[valid]
        icon_weights = icon.expand(N, D, D)[valid]  # [K]
        colors_valid = colors_now[:, :, None, None].expand(N, 3, D, D).permute(1, 0, 2, 3)[
            :, valid
        ]  # [3, K]
        idx_flat = (y_valid * W + x_valid).long()  # [K]

        accum = torch.zeros_like(rgbs[t])  # [3, H, W]
        weight = torch.zeros(1, H * W, device=device)  # [1, H*W]
        img_flat = accum.view(C, -1)  # [3, H*W]
        weighted_colors = colors_valid * icon_weights  # [3, K]
        img_flat.scatter_add_(1, idx_flat.unsqueeze(0).expand(C, -1), weighted_colors)
        weight.scatter_add_(1, idx_flat.unsqueeze(0), icon_weights.unsqueeze(0))
        weight = weight.view(1, H, W)

        alpha = weight.clamp(0, 1) * opacity
        accum = accum / (weight + 1e-6)  # [3, H, W]
        rgbs[t] = rgbs[t] * (1 - alpha) + accum * alpha
    rgbs = rgbs.clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy() # T,H,W,3
    if bkg_opacity==0.0:
        for t in range(T):
            hsv_frame = cv2.cvtColor(rgbs[t], cv2.COLOR_RGB2HSV)
            saturation_factor = 1.5
            hsv_frame[..., 1] = np.clip(hsv_frame[..., 1] * saturation_factor, 0, 255)
            rgbs[t] = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2RGB)
    return rgbs

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        if param > 100000:
            table.add_row([name, param])
        total_params+=param
    print(table)
    print('total params: %.2f M' % (total_params/1000000.0))
    return total_params


def forward_video(rgbs, framerate, model, args):
    
    B,T,C,H,W = rgbs.shape
    assert C == 3
    device = rgbs.device
    assert(B==1)

    grid_xy = utils.basic.gridcloud2d(1, H, W, norm=False, device='cuda:0').float() # 1,H*W,2
    grid_xy = grid_xy.permute(0,2,1).reshape(1,1,2,H,W) # 1,1,2,H,W

    torch.cuda.empty_cache()
    print('starting forward...')
    f_start_time = time.time()

    flows_e, visconf_maps_e, _, _ = \
        model.forward_sliding(rgbs[:, args.query_frame:], iters=args.inference_iters, sw=None, is_training=False)
    traj_maps_e = flows_e.cuda() + grid_xy # B,Tf,2,H,W
    if args.query_frame > 0:
        backward_flows_e, backward_visconf_maps_e, _, _ = \
            model.forward_sliding(rgbs[:, :args.query_frame+1].flip([1]), iters=args.inference_iters, sw=None, is_training=False)
        backward_traj_maps_e = backward_flows_e.cuda() + grid_xy # B,Tb,2,H,W, reversed
        backward_traj_maps_e = backward_traj_maps_e.flip([1])[:, :-1] # flip time and drop the overlapped frame
        backward_visconf_maps_e = backward_visconf_maps_e.flip([1])[:, :-1] # flip time and drop the overlapped frame
        traj_maps_e = torch.cat([backward_traj_maps_e, traj_maps_e], dim=1) # B,T,2,H,W
        visconf_maps_e = torch.cat([backward_visconf_maps_e, visconf_maps_e], dim=1) # B,T,2,H,W
    ftime = time.time()-f_start_time
    print('finished forward; %.2f seconds / %d frames; %d fps' % (ftime, T, round(T/ftime)))
    utils.basic.print_stats('traj_maps_e', traj_maps_e)
    utils.basic.print_stats('visconf_maps_e', visconf_maps_e)

    # subsample to make the vis more readable
    rate = args.rate
    trajs_e = traj_maps_e[:,:,:,::rate,::rate].reshape(B,T,2,-1).permute(0,1,3,2) # B,T,N,2
    visconfs_e = visconf_maps_e[:,:,:,::rate,::rate].reshape(B,T,2,-1).permute(0,1,3,2) # B,T,N,2


    # masking
    if hasattr(args, 'mask') and args.mask is not None:
        # Mask tensor should already be B,H,W or H,W on GPU
        mask_img = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        mask_img = cv2.resize(mask_img, (W, H), interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.from_numpy(mask_img > 0).cuda()  # H x W, boolean

        # Use first frame to get trajectory indices
        xy0 = trajs_e[0,0].cpu().numpy()  # N x 2
        x_idx = np.clip(xy0[:,0].astype(int), 0, W-1)
        y_idx = np.clip(xy0[:,1].astype(int), 0, H-1)
        
        inside_mask = mask_tensor[y_idx, x_idx].cpu().numpy()  # N boolean
        trajs_e = trajs_e[:, :, inside_mask, :]
        visconfs_e = visconfs_e[:, :, inside_mask]
#------


    xy0 = trajs_e[0,0].cpu().numpy()
    colors = utils.improc.get_2d_colors(xy0, H, W)

    fn = args.mp4_path.split('/')[-1].split('.')[0]
    rgb_out_f = './pt_vis_%s_rate%d_q%d.mp4' % (fn, rate, args.query_frame)
    print('rgb_out_f', rgb_out_f)
    temp_dir = 'temp_pt_vis_%s_rate%d_q%d' % (fn, rate, args.query_frame)
    utils.basic.mkdir(temp_dir)
    vis = []

    frames = draw_pts_gpu(rgbs[0].to('cuda:0'), trajs_e[0], visconfs_e[0,:,:,1] > args.conf_thr,
                          colors, rate=rate, bkg_opacity=args.bkg_opacity)
    print('frames', frames.shape)

    if args.vstack:
        frames_top = rgbs[0].clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy() # T,H,W,3
        frames = np.concatenate([frames_top, frames], axis=1)
    elif args.hstack:
        frames_left = rgbs[0].clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy() # T,H,W,3
        frames = np.concatenate([frames_left, frames], axis=2)
    
    
    #naming of img is here
    print('writing frames to disk')
    f_start_time = time.time()
    for ti in range(T):
        temp_out_f = '%s/%03d.jpg' % (temp_dir, ti+40) #+20 means start naming at 020 
        im = PIL.Image.fromarray(frames[ti])
        im.save(temp_out_f)#, "PNG", subsampling=0, quality=80)
    ftime = time.time()-f_start_time
    print('finished writing; %.2f seconds / %d frames; %d fps' % (ftime, T, round(T/ftime)))
    print('writing mp4')
    os.system('/usr/bin/ffmpeg -y -hide_banner -loglevel error -f image2 -framerate %d -pattern_type glob -i "./%s/*.jpg" -c:v libx264 -crf 20 -pix_fmt yuv420p %s' % (framerate, temp_dir, rgb_out_f))







# for saving fmap to .npy use this else, keep use the other one
# def forward_video(rgbs, framerate, model, args):
#     import os, numpy as np, time, torch, PIL
#     from PIL import Image

#     B, T, C, H, W = rgbs.shape
#     assert C == 3
#     device = rgbs.device
#     assert(B == 1)

#     grid_xy = utils.basic.gridcloud2d(1, H, W, norm=False, device=device).float()
#     grid_xy = grid_xy.permute(0, 2, 1).reshape(1, 1, 2, H, W)

#     torch.cuda.empty_cache()
#     print('starting forward...')
#     f_start_time = time.time()

#     os.makedirs("feature_maps", exist_ok=True)

#     with torch.no_grad():
#         # reshape B,T,C,H,W -> (B*T, C, H, W)
#         rgbs_reshape = rgbs.view(B * T, C, H, W)

#         # get feature maps (expects 4D input)
#         fmaps = model.get_fmaps(
#             rgbs_reshape,
#             B=B * T,
#             T=1,
#             sw=None,
#             is_training=False
#         )

#         # reshape back to (B, T, C, Hf, Wf)
#         if fmaps.ndim == 4:
#             C_f, Hf, Wf = fmaps.shape[1], fmaps.shape[2], fmaps.shape[3]
#             fmaps = fmaps.view(B, T, C_f, Hf, Wf)
#         else:
#             print(f"Unexpected feature map shape: {fmaps.shape}")

#         # save each frame's fmap as .npy
#         for t in range(T):
#             np.save(f"feature_maps/frame_{t:04d}.npy", fmaps[0, t].cpu().numpy())
#     print(f"Saved {T} feature maps to feature_maps/")

#     flows_e, visconf_maps_e, _, _ = model.forward_sliding(
#         rgbs[:, args.query_frame:], iters=args.inference_iters,
#         sw=None, is_training=False
#     )

#     traj_maps_e = flows_e.cuda() + grid_xy
#     if args.query_frame > 0:
#         backward_flows_e, backward_visconf_maps_e, _, _ = model.forward_sliding(
#             rgbs[:, :args.query_frame+1].flip([1]),
#             iters=args.inference_iters, sw=None, is_training=False
#         )
#         backward_traj_maps_e = backward_flows_e.cuda() + grid_xy
#         backward_traj_maps_e = backward_traj_maps_e.flip([1])[:, :-1]
#         backward_visconf_maps_e = backward_visconf_maps_e.flip([1])[:, :-1]
#         traj_maps_e = torch.cat([backward_traj_maps_e, traj_maps_e], dim=1)
#         visconf_maps_e = torch.cat([backward_visconf_maps_e, visconf_maps_e], dim=1)

#     ftime = time.time() - f_start_time
#     print(f'finished forward; {ftime:.2f}s / {T} frames; {round(T/ftime)} fps')
#     utils.basic.print_stats('traj_maps_e', traj_maps_e)
#     utils.basic.print_stats('visconf_maps_e', visconf_maps_e)

#     # visualization
#     rate = args.rate
#     trajs_e = traj_maps_e[:,:,:,::rate,::rate].reshape(B, T, 2, -1).permute(0,1,3,2)
#     visconfs_e = visconf_maps_e[:,:,:,::rate,::rate].reshape(B, T, 2, -1).permute(0,1,3,2)
#     xy0 = trajs_e[0,0].cpu().numpy()
#     colors = utils.improc.get_2d_colors(xy0, H, W)

#     fn = args.mp4_path.split('/')[-1].split('.')[0]
#     rgb_out_f = f'./pt_vis_{fn}_rate{rate}_q{args.query_frame}.mp4'
#     print('rgb_out_f', rgb_out_f)
#     temp_dir = f'temp_pt_vis_{fn}_rate{rate}_q{args.query_frame}'
#     utils.basic.mkdir(temp_dir)

#     frames = draw_pts_gpu(
#         rgbs[0].to(device),
#         trajs_e[0],
#         visconfs_e[0,:,:,1] > args.conf_thr,
#         colors,
#         rate=rate,
#         bkg_opacity=args.bkg_opacity
#     )

#     if args.vstack:
#         frames_top = rgbs[0].clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
#         frames = np.concatenate([frames_top, frames], axis=1)
#     elif args.hstack:
#         frames_left = rgbs[0].clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
#         frames = np.concatenate([frames_left, frames], axis=2)

#     print('writing frames to disk')
#     f_start_time = time.time()
#     for ti in range(T):
#         temp_out_f = f'{temp_dir}/{ti:03d}.jpg'
#         Image.fromarray(frames[ti]).save(temp_out_f)
#     ftime = time.time() - f_start_time
#     print(f'finished writing; {ftime:.2f}s / {T} frames; {round(T/ftime)} fps')

#     print('writing mp4')
#     os.system(
#         f'/usr/bin/ffmpeg -y -hide_banner -loglevel error -f image2 '
#         f'-framerate {framerate} -pattern_type glob -i "./{temp_dir}/*.jpg" '
#         f'-c:v libx264 -crf 20 -pix_fmt yuv420p {rgb_out_f}'
#     )
#     # flow vis
#     rgb_out_f = './flow_vis.mp4'
#     temp_dir = 'temp_flow_vis'
#     utils.basic.mkdir(temp_dir)
#     vis = []
#     for ti in range(T):
#         flow_vis = utils.improc.flow2color(flows_e[0:1,ti])
#         vis.append(flow_vis)
#     for ti in range(T):
#         temp_out_f = '%s/%03d.png' % (temp_dir, ti)
#         im = PIL.Image.fromarray(vis[ti][0].permute(1,2,0).cpu().numpy())
#         im.save(temp_out_f, "PNG", subsampling=0, quality=100)
#     os.system('/usr/bin/ffmpeg -y -hide_banner -loglevel error -f image2 -framerate 24 -pattern_type glob -i "./%s/*.png" -c:v libx264 -crf 1 -pix_fmt yuv420p %s' % (temp_dir, rgb_out_f))
    
#     return None


def read_image_folder(folder_path, image_size=1024, max_frames=None):
    import glob
    import cv2
    from PIL import Image
    import torch

    img_paths = sorted(glob.glob(folder_path + '/*.[jp][pn]g'))  # jpg/png

    # Only take images 11-20 (10:20)
    img_paths = img_paths[0:10]

    if max_frames:
        img_paths = img_paths[:max_frames]

    imgs = []
    for path in img_paths:
        img = np.array(Image.open(path).convert('RGB'))
        imgs.append(img)

    H, W = imgs[0].shape[:2]
    scale = min(int(image_size)/H, int(image_size)/W)
    H, W = int(H*scale), int(W*scale)
    H, W = H//8*8, W//8*8

    imgs_resized = [cv2.resize(img, (W,H), interpolation=cv2.INTER_LINEAR) for img in imgs]
    rgbs = [torch.from_numpy(img).permute(2,0,1) for img in imgs_resized]
    rgbs = torch.stack(rgbs, dim=0).unsqueeze(0).float()  # 1,T,C,H,W
    framerate = 24  # dummy, only used if you want to write mp4
    return rgbs, framerate






#for vid
# def run(model, args):
#     log_dir = './logs_demo'
    
#     global_step = 0

#     if args.ckpt_init:
#         _ = utils.saveload.load(
#             None,
#             args.ckpt_init,
#             model,
#             optimizer=None,
#             scheduler=None,
#             ignore_load=None,
#             strict=True,
#             verbose=False,
#             weights_only=False,
#         )
#         print('loaded weights from', args.ckpt_init)
#     else:
#         if args.tiny:
#             url = "https://huggingface.co/aharley/alltracker/resolve/main/alltracker_tiny.pth"
#         else:
#             url = "https://huggingface.co/aharley/alltracker/resolve/main/alltracker.pth"
#         state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu')
#         model.load_state_dict(state_dict['model'], strict=True)
#         print('loaded weights from', url)

#     model.cuda()
#     for n, p in model.named_parameters():
#         p.requires_grad = False
#     model.eval()

#     # uncomment for vid: 
#     rgbs, framerate = read_mp4(args.mp4_path)
#     # if hasattr(args, 'image_folder') and args.image_folder:
#     #     rgbs, framerate = read_image_folder(args.image_folder, image_size=args.image_size, max_frames=args.max_frames)
#     # else:
#     # # original video reading
#     # # rgbs, framerate = read_mp4(args.mp4_path)
#     #     pass

#     print('rgbs[0]', rgbs[0].shape)
#     H,W = rgbs[0].shape[:2]
    
#     # shorten & shrink the video, in case the gpu is small
#     if args.max_frames:
#         rgbs = rgbs[:args.max_frames]
#     scale = min(int(args.image_size)/H, int(args.image_size)/W)
#     H, W = int(H*scale), int(W*scale)
#     H, W = H//8 * 8, W//8 * 8 # make it divisible by 8
#     rgbs = [cv2.resize(rgb, dsize=(W, H), interpolation=cv2.INTER_LINEAR) for rgb in rgbs]
#     print('rgbs[0]', rgbs[0].shape)

#     # move to gpu
#     rgbs = [torch.from_numpy(rgb).permute(2,0,1) for rgb in rgbs]
#     rgbs = torch.stack(rgbs, dim=0).unsqueeze(0).float() # 1,T,C,H,W
#     print('rgbs', rgbs.shape)
    
#     with torch.no_grad():
#         metrics = forward_video(rgbs, framerate, model, args)
    
#     return None


# for img
def run(model, args):

    log_dir = './logs_demo'
    global_step = 0

    # Load checkpoint or default weights
    if args.ckpt_init:
        _ = utils.saveload.load(
            None,
            args.ckpt_init,
            model,
            optimizer=None,
            scheduler=None,
            ignore_load=None,
            strict=True,
            verbose=False,
            weights_only=False,
        )
        print('Loaded weights from', args.ckpt_init)
    else:
        if args.tiny:
            url = "https://huggingface.co/aharley/alltracker/resolve/main/alltracker_tiny.pth"
        else:
            url = "https://huggingface.co/aharley/alltracker/resolve/main/alltracker.pth"
        state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu')
        model.load_state_dict(state_dict['model'], strict=True)
        print('Loaded weights from', url)

    model.cuda()
    for n, p in model.named_parameters():
        p.requires_grad = False
    model.eval()

    # --- Load input frames ---
    if hasattr(args, 'image_folder') and args.image_folder:
        rgbs, framerate = read_image_folder(
            args.image_folder, image_size=args.image_size, max_frames=args.max_frames
        )
    else:
        rgbs, framerate = read_mp4(args.mp4_path)
        # Convert to torch tensor and permute
        rgbs = [torch.from_numpy(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).permute(2,0,1) for f in rgbs]
        rgbs = torch.stack(rgbs, dim=0).unsqueeze(0).float()  # 1,T,C,H,W

    # Move to GPU
    rgbs = rgbs.cuda()
    print('rgbs', rgbs.shape)  # Should be [1, T, 3, H, W]
    
    #  Apply mask 
    mask_resized = None
    if args.mask is not None:
        mask_img = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            raise FileNotFoundError(f"Cannot read mask at {args.mask}")
        
        B, T, C, H, W = rgbs.shape  # fix unpacking
        mask_resized = cv2.resize(mask_img, (W, H), interpolation=cv2.INTER_NEAREST)
        mask_resized = (mask_resized > 0).astype(np.uint8)
        
        mask_tensor = torch.from_numpy(mask_resized).unsqueeze(0).unsqueeze(0).float().cuda()  # 1,1,H,W
        rgbs = rgbs * mask_tensor


    with torch.no_grad():
        metrics = forward_video(rgbs, framerate, model, args)

    return None



def load_mask(mask_path, frame_shape):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask file at {mask_path}")
    if mask.shape != frame_shape[:2]:
        mask = cv2.resize(mask, (frame_shape[1], frame_shape[0]), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)  # binary mask





if __name__ == "__main__":
    torch.set_grad_enabled(False)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_init", type=str, default='') # the ckpt we want (else default)
    parser.add_argument("--mp4_path", type=str, default='./demo_video/monkey.mp4') # input video 
    parser.add_argument("--query_frame", type=int, default=0) # which frame to track from
    parser.add_argument("--image_size", type=int, default=1024) # max dimension of a video frame (upsample to this)
    parser.add_argument("--max_frames", type=int, default=400) # trim the video to this length
    parser.add_argument("--inference_iters", type=int, default=4) # number of inference steps per forward
    parser.add_argument("--window_len", type=int, default=16) # model hyperparam
    parser.add_argument("--rate", type=int, default=2) # vis hyp
    parser.add_argument("--conf_thr", type=float, default=0.1) # vis hyp
    parser.add_argument("--bkg_opacity", type=float, default=0.5) # vis hyp
    parser.add_argument("--vstack", action='store_true', default=False) # whether to stack the input and output in the mp4
    parser.add_argument("--hstack", action='store_true', default=False) # whether to stack the input and output in the mp4
    parser.add_argument("--tiny", action='store_true', default=False) # whether to use the tiny model
    parser.add_argument("--image_folder", type=str, default='')  # folder of images (comment for vid run)
    parser.add_argument("--mask", type=str, default=None) # add masking to images

    args = parser.parse_args()

    from nets.alltracker import Net;
    if args.tiny:
        model = Net(args.window_len, use_basicencoder=True, no_split=True)
    else:
        model = Net(args.window_len)
    count_parameters(model)

    run(model, args)
    












































# def test():
    #---------------------------IMPORTS------------------------------
    import os
    import sys
    import cv2
    import numpy as np
    import torch
    from types import SimpleNamespace
    import sqlite3
    #from Database.visualize import visualize_from_db




    #---------------------------CONFIG------------------------------
    sys.path.append(os.path.abspath("./all_t_git"))
    # Import alltracker after path is set
    from nets.alltracker import Net
    from demo import run, read_image_folder, forward_video

    # Import database module
    from Database.Db import COLMAPDatabase

    #---------------------------DIRECTORIES-------------------------
    # "./All_t/masked"
    RAW_IMG_DIR = "test_sin"
    DB_PATH = "./Database/alltracker.db"
    VIS_OUT_DIR = "./Database/vis"
    os.makedirs(VIS_OUT_DIR, exist_ok=True)
    MASK_PATH = "/mnt/data1/michelle/t_alltracker/All_t/undistorted_mask.bmp"




    #---------------------------ALLTRACKER-----------------------------
    # def run_alltracker(model, input_dir, args):
    #     # Load images
    #     rgbs, framerate = read_image_folder(
    #         input_dir, image_size=args.image_size, max_frames=args.max_frames
    #     )

    #     if len(rgbs) == 0:
    #         print("[AllTracker] No images found in", input_dir)
    #         return {}

    #     # Forward through AllTracker
    #     print("[AllTracker] Running forward pass")
    #     with torch.no_grad():
    #         traj_maps, vis_maps, *_ = model.forward_sliding(
    #             rgbs, iters=args.inference_iters, window_len=args.window_len
    #         )

    #     # Convert output to per-frame keypoints
    #     results = {}
    #     num_frames = traj_maps.shape[1]  # T
    #     for t in range(num_frames):
    #         traj_frame = traj_maps[0, t].cpu().numpy()  # HxW
    #         yx = np.argwhere(traj_frame > 0)  # list of (y, x)
    #         keypoints = yx[:, ::-1]  # convert to (x, y)
    #         descriptors = np.ones((len(keypoints), 32), dtype=np.float32)  # placeholder
    #         results[f"frame_{t:04d}.png"] = (keypoints, descriptors)

    #     print(f"[AllTracker] Extracted keypoints for {len(results)} frames.")
    #     return results

    def run_alltracker(model, input_dir, args):
        rgbs, framerate = read_image_folder(
            input_dir, image_size=args.image_size, max_frames=args.max_frames
        )
        if len(rgbs) == 0:
            print("[AllTracker] No images found in", input_dir)
            return {}, {}
        # Move to GPU and eval
        rgbs = rgbs.cuda()
        for n, p in model.named_parameters():
            p.requires_grad = False
        model.eval()
        # Forward through AllTracker (same as demo)
        print("[AllTracker] Running forward pass")
        with torch.no_grad():
            traj_maps, vis_maps, *_ = model.forward_sliding(
                rgbs, iters=args.inference_iters, window_len=args.window_len
            )
        # Convert outputs
        traj_maps = traj_maps[0].cpu().numpy()   # (T, 2, H, W)
        vis_maps  = vis_maps[0].cpu().numpy()    # (T, 1, H, W)
        T, _, H, W = traj_maps.shape
        xs, ys = np.meshgrid(np.arange(W), np.arange(H))
        base_pts = np.stack([xs.flatten(), ys.flatten()], axis=-1).astype(np.float32)  # (H*W, 2)
        results = {}
        correspondences = {}

        for t in range(T):
            flow = traj_maps[t]        # (2, H, W)
            conf = vis_maps[t, 0]      # (H, W)
            mask = conf > args.conf_thr

            # keypoints in frame t (x,y)
            ft_xy = np.stack([xs[mask], ys[mask]], axis=1).astype(np.float32)

            # corresponding coordinates
            flow_x = flow[0]
            flow_y = flow[1]
            f0_xy = np.stack([
                xs[mask] - flow_x[mask],
                ys[mask] - flow_y[mask]
            ], axis=1).astype(np.float32)

            # indices into flattened base_pts
            idx0 = np.flatnonzero(mask.flatten()).astype(np.uint32)
            idxt = np.arange(len(ft_xy), dtype=np.uint32)

            # Save keypoints (descriptors r empty)
            fname = f"frame_{t:04d}.png"
            results[fname] = (ft_xy, np.zeros((len(ft_xy), 32), dtype=np.uint8))

            # Save correspondence
            correspondences[t] = {
                "idx0": idx0,
                "idxt": idxt,
                "f0_xy": f0_xy,
                "ft_xy": ft_xy,
                "mask": mask  # keep 2D
            }


        print(f"[AllTracker] finished.")
        return results, correspondences


    def img_resize(img, image_size=1024): #same as alltracker read_image_folder
        import cv2
        H0, W0 = img.shape[:2]
        scale = min(image_size / H0, image_size / W0)
        H = int(H0 * scale)
        W = int(W0 * scale)
        H = (H // 8) * 8
        W = (W // 8) * 8
        img_resized = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
        return img_resized





    #---------------------------ORB TRACKING------------------------------
    #Use ORB to refine and track keypoints between consecutive frames.
    # def orb_track(results, input_dir, max_kp=5000):
    #     orb = cv2.ORB_create(nfeatures=1000)
    #     tracked_results = {}
    #     filenames = sorted(list(results.keys()))
    #     prev_kps, prev_desc = None, None

    #     for i, filename in enumerate(filenames):
    #         img_path = os.path.join(input_dir, filename)
    #         img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    #         if img is None:
    #             print(f"[ORB Track] WARNING: Could not read {filename}, skipping")
    #             continue
    #         img = img_resize(img)
    #         print(f"[ORB Track] Processing {filename}")
    #         kps_all, desc_all = results[filename]
    #         # Convert keypoints (x,y) to cv2.KeyPoint objects
    #         kps_all_cv2 = [cv2.KeyPoint(float(k[0]), float(k[1]), 1) for k in kps_all] if len(kps_all) > 0 else []
    #         # SUBSAMPLE since theres too many matches, orb cant process all 
    #         if len(kps_all_cv2) > max_kp:
    #             idxs = np.linspace(0, len(kps_all_cv2) - 1, max_kp).astype(int)
    #             kps_all_cv2 = [kps_all_cv2[idx] for idx in idxs]
    #             print(f"[ORB Track] Subsampled keypoints to {max_kp} (from {len(idxs)})")
    #         # Compute ORB descriptors (will return list of KeyPoints + descriptors or (None, None))
    #         if len(kps_all_cv2) > 0:
    #             kps_refined, desc_refined = orb.compute(img, kps_all_cv2)
    #             if desc_refined is None:
    #                 # ensure we always store a 2D uint8 descriptor array (possibly empty)
    #                 desc_refined = np.zeros((0, 32), dtype=np.uint8)
    #                 kps_refined = []
    #             else:
    #                 desc_refined = np.asarray(desc_refined, dtype=np.uint8)
    #                 if desc_refined.ndim == 1:
    #                     desc_refined = desc_refined.reshape(1, -1)
    #         else:
    #             kps_refined, desc_refined = [], np.zeros((0, 32), dtype=np.uint8)

    #         # Debug info
    #         #print(f"[ORB Track DEBUG] prev_desc: {None if prev_desc is None else (type(prev_desc), getattr(prev_desc,'shape',None), getattr(prev_desc,'dtype',None))}")
    #         #print(f"[ORB Track DEBUG] desc_refined: {(type(desc_refined), desc_refined.shape, desc_refined.dtype)}")

    #         # Match if both descriptor sets are non-empty 2D arrays
    #         if (prev_desc is not None and isinstance(prev_desc, np.ndarray) and prev_desc.size > 0 and
    #             isinstance(desc_refined, np.ndarray) and desc_refined.size > 0):
    #             prev_desc = np.asarray(prev_desc, dtype=np.uint8)
    #             desc_refined = np.asarray(desc_refined, dtype=np.uint8)
    #             if prev_desc.ndim != 2 or desc_refined.ndim != 2:
    #                 print(f"[ORB Track] Unexpected descriptor ndim: prev {prev_desc.ndim}, cur {desc_refined.ndim}. Skipping matching.")
    #             else:
    #                 try:
    #                     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    #                     matches = bf.match(prev_desc, desc_refined)
    #                     matches = sorted(matches, key=lambda x: x.distance)
    #                     print(f"[ORB Track] {len(matches)} matches between frames {filenames[i-1]} and {filename}")
    #                 except cv2.error as e:
    #                     print(f"[ORB Track] OpenCV matcher error: {e}. Skipping matching for this pair.")
    #         else:
    #             print(f"[ORB Track] Skipping matching for frame {filename} (no descriptors)")
    #         # Save results: convert KeyPoints to Nx2 float32 points; keep descriptors as ndarray Nx32 uint8
    #         pts = (np.array([[kp.pt[0], kp.pt[1]] for kp in kps_refined], dtype=np.float32)
    #                if kps_refined else np.zeros((0, 2), dtype=np.float32))
    #         tracked_results[filename] = (pts, desc_refined if isinstance(desc_refined, np.ndarray) else np.zeros((0, 32), dtype=np.uint8))
    #         prev_kps, prev_desc = kps_refined, desc_refined
    #     return tracked_results


    #---------------------------SAVE TO DB------------------------------
    # return Nx6 array for COLMAP
    def normalize_to_colmap_kp(pts):
        if pts.shape[1] == 2:  
            N = pts.shape[0]
            scales = np.ones((N, 1), np.float32)
            orientations = np.zeros((N, 1), np.float32)
            a4 = np.zeros((N, 1), np.float32)
            scores = np.ones((N, 1), np.float32)
            return np.hstack([pts, scales, orientations, a4, scores]).astype(np.float32)
        return pts.astype(np.float32)




    # filtering keypoints to demo the matching
    def filter_keypoints(pts, desc=None, max_kps=2000):
        if pts is None or len(pts) == 0:
            return pts, None, np.array([], dtype=np.int32)
        N, C = pts.shape
        if N <= max_kps:
            return pts.astype(np.float32), desc, np.arange(N)
        if C > 2:  # has scores
            scores = pts[:, 2]
            sel_idx = np.argsort(scores)[-max_kps:] # best ones
        else:
            sel_idx = np.random.choice(N, max_kps, replace=False)
        pts_f = pts[sel_idx].astype(np.float32)
        
        # handling descr = none
        if desc is not None and desc.shape[0] > 0:
            desc_f = desc[sel_idx]
        else:
            desc_f = None

        return pts_f, desc_f, sel_idx

    # saving to db
    def save_to_database(db_path, results_and_corr, max_kps=2000): # max keypoints
        results, correspondences = results_and_corr
        db = COLMAPDatabase.connect(db_path)
        db.create_tables()
        image_id_map = {}
        keypoints_map = {}  # for matching
        
        # Saving keypts
        for filename, (pts, desc) in results.items():
            # filter keypoints
            pts_f, desc_f, sel_idx = filter_keypoints(pts, desc, max_kps=max_kps)
            keypoints_map[filename] = sel_idx
            # convert to COLMAP format
            colmap_kps = normalize_to_colmap_kp(pts_f)
            # create img
            try:
                db.add_image(name=filename, camera_id=1)
            except sqlite3.IntegrityError:
                pass
            image_id = db.execute(
                "SELECT image_id FROM images WHERE name=?", (filename,)
            ).fetchone()[0]
            image_id_map[filename] = image_id
            # save keypoints
            db.execute("DELETE FROM keypoints WHERE image_id=?", (image_id,))
            if colmap_kps.shape[0] > 0:
                db.add_keypoints(image_id, colmap_kps)
                print(f"[DB] Added {colmap_kps.shape[0]} keypoints for {filename}")
            else:
                print(f"[DB] No keypoints for {filename}")
            # save descriptors
            db.execute("DELETE FROM descriptors WHERE image_id=?", (image_id,))
            if desc_f is not None and desc_f.shape[0] > 0:
                db.add_descriptors(image_id, desc_f.astype(np.uint8))

        # save matches (for fmap correspondance)
        id0 = image_id_map.get("frame_0000.png", None)
        sel0 = keypoints_map.get("frame_0000.png", None)
        if id0 is None or sel0 is None:
            print("[DB] ERROR: frame_0000.png missing")
        else:
            for t, corr in correspondences.items():
                fname = f"frame_{t:04d}.png"
                idt = image_id_map.get(fname, None)
                selt = keypoints_map.get(fname, None)
                if idt is None or selt is None:
                    continue
                idx0, idxt = corr["idx0"].astype(np.uint32), corr["idxt"].astype(np.uint32)
                # keep only matches with filtered keypoints
                mask0 = np.isin(idx0, sel0)
                maskt = np.isin(idxt, selt)
                mask = mask0 & maskt
                if not np.any(mask):
                    continue
                # remap
                idx_map0 = {old: new for new, old in enumerate(sel0)}
                idx_mapt = {old: new for new, old in enumerate(selt)}
                idx0_f = np.array([idx_map0[i] for i in idx0[mask]], dtype=np.uint32)
                idxt_f = np.array([idx_mapt[i] for i in idxt[mask]], dtype=np.uint32)

                match_arr = np.stack([idx0_f, idxt_f], axis=1)
                pair_id = (id0 << 32) + idt
                db.execute(
                    "INSERT OR REPLACE INTO matches(pair_id, rows, cols, data) VALUES (?,?,?,?)",
                    (pair_id, match_arr.shape[0], match_arr.shape[1], match_arr.tobytes())
                )
                print(f"[DB] Added {match_arr.shape[0]} matches 0 to {fname}")
        db.commit()
        db.close()
        print("[DB] Finished saving keypoints + matches.")


    def apply_mask_to_image(img, mask):
        mask = (mask > 0).astype(np.uint8)
        out = img.copy()
        out[mask == 0] = 0
        return out

    # ------------------VISUALIZE DB------------------
    # visualzing database by plotting out keypoints
    # def visualize_from_db_with_mask(db_path, img_dir, mask_path, out_dir, image_size=1024):
    #     """
    #     Visualize keypoints from DB (green dots) on images, then overlay mask.
    #     """
    #     os.makedirs(out_dir, exist_ok=True)

    #     # Load mask once
    #     mask_orig = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    #     assert mask_orig is not None, f"Mask not found: {mask_path}"

    #     db = COLMAPDatabase.connect(db_path)

    #     # Map image_id -> filename
    #     rows = db.execute("SELECT image_id, name FROM images").fetchall()
    #     image_map = {row[0]: row[1] for row in rows}

    #     # Load all keypoints from DB
    #     keypoints_dict = db.read_all_keypoints()  # {image_id: Nx6 array}

    #     for image_id, kps in keypoints_dict.items():
    #         if kps is None or len(kps) == 0:
    #             continue

    #         filename = image_map.get(image_id, None)
    #         if filename is None:
    #             continue

    #         img_path = os.path.join(img_dir, filename)
    #         img = cv2.imread(img_path)
    #         if img is None:
    #             continue

    #         # Resize image like AllTracker
    #         H0, W0 = img.shape[:2]
    #         scale = min(image_size / H0, image_size / W0)
    #         H = int(H0 * scale)
    #         W = int(W0 * scale)
    #         H = (H // 8) * 8
    #         W = (W // 8) * 8
    #         img_r = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)

    #         # Draw keypoints
    #         vis = img_r.copy()
    #         kps_2d = np.array(kps, dtype=np.float32)[:, :2]  # Nx2
    #         for x, y in kps_2d:
    #             cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)

    #         # Resize mask to match resized image
    #         mask_r = cv2.resize(mask_orig, (W, H), interpolation=cv2.INTER_NEAREST)
    #         mask_r = (mask_r > 0).astype(np.uint8)

    #         # Apply mask: pixels outside mask blacked out (keypoints inside mask remain)
    #         vis[mask_r == 0] = 0

    #         # Save visualization
    #         out_path = os.path.join(out_dir, f"vis_{filename}")
    #         cv2.imwrite(out_path, vis)

    #     db.close()
    #     print(f"[VIS] Saved masked keypoint visualizations to {out_dir}")




    def visualize_from_db(db_path, img_dir, out_dir, image_size=1024):
        """
        Visualize keypoints stored in the database on the original images.
        """
        os.makedirs(out_dir, exist_ok=True)

        db = COLMAPDatabase.connect(db_path)

        # Map image_id → filename
        rows = db.execute("SELECT image_id, name FROM images").fetchall()
        image_map = {row[0]: row[1] for row in rows}

        # Load all keypoints
        keypoints_dict = db.read_all_keypoints()

        # Helper to resize images like AllTracker
        def preprocess_img(img):
            H0, W0 = img.shape[:2]
            scale = min(image_size / H0, image_size / W0)
            H = int(H0 * scale)
            W = int(W0 * scale)
            H = (H // 8) * 8
            W = (W // 8) * 8
            return cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)

        for image_id, kps in keypoints_dict.items():
            if kps is None or len(kps) == 0:
                continue

            filename = image_map.get(image_id, None)
            if filename is None:
                continue

            img_path = os.path.join(img_dir, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Resize image
            img_resized = preprocess_img(img)
            vis = img_resized.copy()

            # Draw keypoints
            kps_2d = np.array(kps, dtype=np.float32)[:, :2]  # Nx2
            for x, y in kps_2d:
                cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)

            # Save visualization
            out_path = os.path.join(out_dir, f"vis_{filename}")
            cv2.imwrite(out_path, vis)

        db.close()
        print("[VIS] Done.")



    def resize_like_alltracker(img, image_size=1024):
        H0, W0 = img.shape[:2]
        scale = min(image_size / H0, image_size / W0)
        H = int(H0 * scale)
        W = int(W0 * scale)
        H = (H // 8) * 8
        W = (W // 8) * 8
        img_r = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
        return img_r, (H0, W0), (H, W)


    def undo_resize_pts(pts, orig_shape, resized_shape):
        H0, W0 = orig_shape
        Hr, Wr = resized_shape
        sx = W0 / Wr
        sy = H0 / Hr
        pts = pts.copy()
        pts[:, 0] *= sx
        pts[:, 1] *= sy
        return pts


    def visualize_masked_images(img_dir, mask_path, out_dir, image_size=1024):
        os.makedirs(out_dir, exist_ok=True)

        mask_orig = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_orig is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            img = cv2.imread(os.path.join(img_dir, fname))
            if img is None:
                continue

            img_r, _, _ = resize_like_alltracker(img, image_size)
            mask_r = cv2.resize(
                mask_orig,
                (img_r.shape[1], img_r.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
            mask_r = (mask_r > 0).astype(np.uint8)

            masked = apply_mask_to_image(img_r, mask_r)
            cv2.imwrite(os.path.join(out_dir, f"masked_{fname}"), masked)

        print(f"[VIS] Saved masked images to {out_dir}")















    #---------------------------FEATURE CORRESPONDANCE------------------------------
    def f_matches(
        img0_path,
        img1_path,
        kps0,
        kps1,
        matches,
        mask_path=None,
        max_vis=2000,
        image_size=1024,
        out_path="vis_matches.png"
    ):
        # ---- Load images ----
        img0 = cv2.imread(img0_path)
        img1 = cv2.imread(img1_path)
        assert img0 is not None and img1 is not None

        # ---- Load mask (image-space) ----
        mask = None
        if mask_path is not None and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask = (mask > 0).astype(np.uint8)

        # ---- Resize like AllTracker ----
        img0_r, orig0, resized = resize_like_alltracker(img0, image_size)
        img1_r, orig1, _ = resize_like_alltracker(img1, image_size)

        # ---- Resize mask to match visualization ----
        if mask is not None:
            mask = cv2.resize(mask, (img0_r.shape[1], img0_r.shape[0]))

            # ---- SAVE MASKED IMAGES (debug output) ----
            masked0 = apply_mask_to_image(img0_r, mask)
            masked1 = apply_mask_to_image(img1_r, mask)

            base = os.path.splitext(out_path)[0]
            cv2.imwrite(base + "_masked_ref.png", masked0)
            cv2.imwrite(base + "_masked_tgt.png", masked1)

        else:
            masked0 = img0_r
            masked1 = img1_r

        # ---- Cap visualization ONLY ----
        if matches.shape[0] > max_vis:
            sel = np.random.choice(matches.shape[0], max_vis, replace=False)
            matches = matches[sel]

        # ---- Undo resize for drawing ----
        kps0_draw = undo_resize_pts(kps0[:, :2], orig0, resized)
        kps1_draw = undo_resize_pts(kps1[:, :2], orig1, resized)

        # ---- Draw canvas on MASKED images ----
        h = max(masked0.shape[0], masked1.shape[0])
        w = masked0.shape[1] + masked1.shape[1]
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

        canvas[:masked0.shape[0], :masked0.shape[1]] = masked0
        canvas[:masked1.shape[0], masked0.shape[1]:] = masked1

        for i0, i1 in matches:
            x0, y0 = kps0_draw[i0]
            x1, y1 = kps1_draw[i1]
            x1 += masked0.shape[1]

            cv2.circle(canvas, (int(x0), int(y0)), 2, (0, 255, 0), -1)
            cv2.circle(canvas, (int(x1), int(y1)), 2, (0, 255, 0), -1)
            cv2.line(canvas, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 1)

        cv2.imwrite(out_path, canvas)
        print(f"[VIS] Saved {out_path} ({len(matches)} matches)")

        
        
        

    #---------------------------PIPLINE------------------------------
    # 0. Alltracker model
    def main():
        print("0. Creating AllTracker model")
        args = SimpleNamespace(
            ckpt_init="./checkpoints/alltracker.pth",
            image_folder=RAW_IMG_DIR,
            query_frame=0,
            image_size=1024,
            max_frames=400,
            inference_iters=4,
            window_len=16,
            rate=2,
            conf_thr=0.1,
            bkg_opacity=0.5,
            vstack=False,
            hstack=False,
            tiny=False,
            mask=None
        )

    # 0. DB tables
        MAX_IMAGE_ID = 2**31 - 1
        CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
            camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            model INTEGER NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            params BLOB,
            prior_focal_length INTEGER NOT NULL
        );"""

        CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
            image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            name TEXT NOT NULL UNIQUE,
            camera_id INTEGER NOT NULL,
            CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
            FOREIGN KEY(camera_id) REFERENCES cameras(camera_id)
        );""".format(MAX_IMAGE_ID)

        CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
        image_id INTEGER NOT NULL UNIQUE,
        rows INTEGER NOT NULL,
        cols INTEGER NOT NULL,
        data BLOB,
        FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE);
        """

        CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
        image_id INTEGER NOT NULL UNIQUE,
        rows INTEGER NOT NULL,
        cols INTEGER NOT NULL,
        data BLOB,
        FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE);
        """

        CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
            pair_id INTEGER PRIMARY KEY NOT NULL,
            rows INTEGER NOT NULL,
            cols INTEGER NOT NULL,
            data BLOB
        );"""

        CREATE_TWO_VIEW_GEOMETRIES_TABLE = """CREATE TABLE IF NOT EXISTS two_view_geometries (
            pair_id INTEGER PRIMARY KEY NOT NULL,
            rows INTEGER NOT NULL,
            cols INTEGER NOT NULL,
            data BLOB,
            config INTEGER NOT NULL,
            F BLOB,
            E BLOB,
            H BLOB,
            qvec BLOB,
            tvec BLOB
        );"""

        CREATE_POSE_PRIORS_TABLE = """CREATE TABLE IF NOT EXISTS pose_priors (
            image_id INTEGER PRIMARY KEY NOT NULL,
            position BLOB,
            coordinate_system INTEGER NOT NULL,
            position_covariance BLOB,
            FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE
        );"""

        CREATE_NAME_INDEX = "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name);"

        CREATE_ALL = "; ".join([
            CREATE_CAMERAS_TABLE,
            CREATE_IMAGES_TABLE,
            CREATE_KEYPOINTS_TABLE,
            CREATE_DESCRIPTORS_TABLE,
            CREATE_MATCHES_TABLE,
            CREATE_TWO_VIEW_GEOMETRIES_TABLE,
            CREATE_POSE_PRIORS_TABLE,
            CREATE_NAME_INDEX
        ])
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.executescript(CREATE_ALL)
        conn.commit()
        conn.close()

        # create model
        model = Net(args.window_len)
        model.cuda()
        for p in model.parameters():
            p.requires_grad = False

        # load checkpoint
        if args.ckpt_init:
            checkpoint = torch.load(args.ckpt_init, map_location='cpu')
            model.load_state_dict(checkpoint['model'], strict=True)
            #print(f"[MODEL] Loaded checkpoint from {args.ckpt_init}")
        else:
            print("[MODEL] Using default weights")

    # -------------------RUN ALLTRACKER-------------------------
        print("\n1. Running AllTracker")
        alltracker_results_and_corr = run_alltracker(model, RAW_IMG_DIR, args)

    # -------------------SAVE TO DATABASE-----------------------
        print("\n2. Saving to database")
        save_to_database(DB_PATH, alltracker_results_and_corr)
        
    # -------------------VISUALIZE MASKED IMAGES--------------------
        print("\n3. Visualizing masked images")

        VIS_DIR = os.path.join(os.path.dirname(DB_PATH), "vis")
        visualize_from_db(
            db_path=DB_PATH,
            img_dir=RAW_IMG_DIR,
            # mask_path=MASK_PATH,
            out_dir=VIS_DIR,
            image_size=args.image_size
        )


    # -------------------VISUALIZE FEATURE MATCHES--------------
        print("\n4. Visualizing feature correspondences")

        results, correspondences = alltracker_results_and_corr

        os.makedirs(os.path.join(os.path.dirname(DB_PATH), "matches"), exist_ok=True)

        ref_name = "frame_0000.png"
        kps0, _ = results[ref_name]

        for t, corr in correspondences.items():
            fname = f"frame_{t:04d}.png"
            if fname not in results:
                continue

            kps1, _ = results[fname]

            matches = np.stack(
                [corr["idx0"], corr["idxt"]], axis=1
            ).astype(np.int32)

            MASK_DIR = "/All_t/undistorted_mask.bmp"

            mask_path = os.path.join(MASK_DIR, f"{fname.replace('.png', '.bmp')}")


            # only visualize matches that exist after filtering
            valid = (matches[:,0] < len(kps0)) & (matches[:,1] < len(kps1))
            matches = matches[valid]

            f_matches(
                img0_path=os.path.join(RAW_IMG_DIR, ref_name),
                img1_path=os.path.join(RAW_IMG_DIR, fname),
                kps0=kps0,
                kps1=kps1,
                matches=matches,
                mask_path=mask_path,  
                max_vis=2000,
                out_path=os.path.join(
                    os.path.dirname(DB_PATH),
                    "matches",
                    f"matches_{fname}"
                )
            )




    if __name__ == "__main__":
        main()