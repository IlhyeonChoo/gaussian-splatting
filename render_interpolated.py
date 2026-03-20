import torch
import numpy as np
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scipy.spatial.transform import Rotation, Slerp

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def interpolate_cameras(cam1, cam2, alpha):
    """두 카메라 사이를 보간"""
    # Rotation 보간 (Slerp)
    R1 = cam1.R if isinstance(cam1.R, np.ndarray) else cam1.R.cpu().numpy()
    R2 = cam2.R if isinstance(cam2.R, np.ndarray) else cam2.R.cpu().numpy()
    
    rot1 = Rotation.from_matrix(R1)
    rot2 = Rotation.from_matrix(R2)
    
    slerp = Slerp([0, 1], Rotation.concatenate([rot1, rot2]))
    R_interp = slerp(alpha).as_matrix()
    
    # Translation 보간 (선형)
    T1 = cam1.T if isinstance(cam1.T, np.ndarray) else cam1.T.cpu().numpy()
    T2 = cam2.T if isinstance(cam2.T, np.ndarray) else cam2.T.cpu().numpy()
    T_interp = (1 - alpha) * T1 + alpha * T2
    
    return R_interp, T_interp


def render_interpolated_path(dataset, iteration, pipeline, frames_between=10):
    """학습 카메라들 사이를 보간해서 렌더링"""
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        # 학습 카메라 가져오기
        train_cams = scene.getTrainCameras()
        print(f"Found {len(train_cams)} training cameras")
        
        # 렌더링 경로
        render_path = os.path.join(dataset.model_path, "video_interpolated")
        makedirs(render_path, exist_ok=True)
        
        frame_idx = 0
        
        # 카메라들 사이를 보간
        for i in range(len(train_cams)):
            cam_start = train_cams[i]
            cam_end = train_cams[(i + 1) % len(train_cams)]  # 순환
            
            for j in range(frames_between if i < len(train_cams) - 1 else frames_between + 1):
                alpha = j / frames_between
                
                # 카메라 보간
                R_interp, T_interp = interpolate_cameras(cam_start, cam_end, alpha)
                
                # 더미 이미지 (PIL Image)
                from PIL import Image
                dummy_image = Image.new('RGB', (cam_start.image_width, cam_start.image_height), (0, 0, 0))
                
                # 새 카메라 생성
                from scene.cameras import Camera
                cam = Camera(
                    resolution=(cam_start.image_width, cam_start.image_height),
                    colmap_id=frame_idx,
                    R=R_interp,
                    T=T_interp,
                    FoVx=cam_start.FoVx,
                    FoVy=cam_start.FoVy,
                    depth_params=None,
                    image=dummy_image,
                    invdepthmap=None,
                    image_name=f"frame_{frame_idx:05d}",
                    uid=frame_idx,
                    data_device="cuda"
                )
                
                # 렌더링
                rendering = render(cam, gaussians, pipeline, background, 
                                 use_trained_exp=False, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                
                # 이미지 저장
                torchvision.utils.save_image(rendering, os.path.join(render_path, f'{frame_idx:05d}.png'))
                frame_idx += 1
        
        print(f"\nRendering complete! {frame_idx} frames saved to: {render_path}")
        return render_path


if __name__ == "__main__":
    parser = ArgumentParser(description="Render interpolated camera path")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--frames_between", default=10, type=int, 
                       help="Number of frames to interpolate between each camera pair")
    parser.add_argument("--quiet", action="store_true")
    
    args = get_combined_args(parser)
    print("Rendering interpolated path for: " + args.model_path)
    
    safe_state(args.quiet)
    
    render_path = render_interpolated_path(
        model.extract(args), 
        args.iteration, 
        pipeline.extract(args),
        args.frames_between
    )
    
    print("\nTo create video, run:")
    print(f"ffmpeg -framerate 30 -i {render_path}/%05d.png -c:v libx264 -pix_fmt yuv420p -crf 18 interpolated_video.mp4")
