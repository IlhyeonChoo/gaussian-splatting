import torch
import numpy as np
from scene import Scene
from scene.cameras import Camera
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def create_circular_path(center, radius, height, num_frames=120):
    """원형 카메라 경로 생성"""
    angles = np.linspace(0, 2 * np.pi, num_frames, endpoint=False)
    positions = []
    
    for angle in angles:
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        z = center[2] + height
        positions.append([x, y, z])
    
    return np.array(positions)


def look_at(eye, target, up=np.array([0, 0, 1])):
    """카메라 회전 행렬 계산 (COLMAP convention)"""
    # COLMAP uses: z forward, x right, y down
    z = target - eye
    z = z / np.linalg.norm(z)
    
    x = np.cross(z, up)
    x = x / np.linalg.norm(x)
    
    y = np.cross(z, x)
    
    # COLMAP: camera-to-world rotation
    R = np.stack([x, y, z], axis=1)  # column vectors
    return R.T  # world-to-camera (transpose)


def render_circular_path(dataset, iteration, pipeline, num_frames=120, radius_scale=1.5):
    """원형 경로로 렌더링"""
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        # 학습 카메라에서 중심점 계산
        train_cams = scene.getTrainCameras()
        positions = np.array([cam.camera_center.cpu().numpy() for cam in train_cams])
        center = positions.mean(axis=0)
        
        # 반경 계산
        distances = np.linalg.norm(positions - center, axis=1)
        radius = distances.mean() * radius_scale
        
        # 높이 (z 좌표 평균)
        height = 0.0  # center[2]와 같은 높이
        
        print(f"Center: {center}")
        print(f"Radius: {radius}")
        print(f"Creating {num_frames} frames...")
        
        # 원형 경로 생성
        camera_positions = create_circular_path(center, radius, height, num_frames)
        
        # 렌더링 경로
        render_path = os.path.join(dataset.model_path, "video", f"ours_{iteration}")
        makedirs(render_path, exist_ok=True)
        
        # 첫 번째 학습 카메라 속성 참조
        ref_cam = train_cams[0]
        
        for idx, cam_pos in enumerate(tqdm(camera_positions, desc="Rendering video frames")):
            # Look-at 행렬 계산
            R = look_at(cam_pos, center)
            T = cam_pos
            
            # 더미 이미지 생성 (PIL Image)
            from PIL import Image
            dummy_image = Image.new('RGB', (ref_cam.image_width, ref_cam.image_height), (0, 0, 0))
            
            # 카메라 생성 (R, T는 numpy array로 전달)
            cam = Camera(
                resolution=(ref_cam.image_width, ref_cam.image_height),
                colmap_id=idx,
                R=R,  # numpy array
                T=T,  # numpy array
                FoVx=ref_cam.FoVx,
                FoVy=ref_cam.FoVy,
                depth_params=None,
                image=dummy_image,
                invdepthmap=None,
                image_name=f"frame_{idx:05d}",
                uid=idx,
                data_device="cuda"
            )
            
            # 렌더링
            rendering = render(cam, gaussians, pipeline, background, 
                             use_trained_exp=False, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
            
            # 이미지 저장
            torchvision.utils.save_image(rendering, os.path.join(render_path, f'{idx:05d}.png'))
        
        print(f"\nRendering complete! Frames saved to: {render_path}")
        return render_path


if __name__ == "__main__":
    parser = ArgumentParser(description="Render circular camera path for video")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--num_frames", default=120, type=int, help="Number of frames to render")
    parser.add_argument("--radius_scale", default=1.5, type=float, help="Scale factor for camera radius")
    parser.add_argument("--quiet", action="store_true")
    
    args = get_combined_args(parser)
    print("Rendering video for: " + args.model_path)
    
    safe_state(args.quiet)
    
    render_path = render_circular_path(
        model.extract(args), 
        args.iteration, 
        pipeline.extract(args),
        args.num_frames,
        args.radius_scale
    )
    
    print("\nTo create video, run:")
    print(f"ffmpeg -framerate 30 -i {render_path}/%05d.png -c:v libx264 -pix_fmt yuv420p -crf 18 output_video.mp4")
