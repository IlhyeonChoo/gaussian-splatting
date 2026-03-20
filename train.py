#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim  # 손실 함수들
from gaussian_renderer import render, network_gui  # 렌더링 및 실시간 뷰어
import sys
from scene import Scene, GaussianModel  # 씬과 Gaussian 모델
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm  # 진행률 표시
from utils.image_utils import psnr  # 이미지 품질 측정
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

# TensorBoard (학습 과정 시각화 도구) 사용 가능 여부 확인
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# Fused SSIM (더 빠른 SSIM 계산) 사용 가능 여부 확인
try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

# Sparse Adam (최적화된 옵티마이저) 사용 가능 여부 확인
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    """3D Gaussian Splatting 메인 학습 함수"""
    
    # Sparse Adam 옵션 확인
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)  # 출력 폴더 및 TensorBoard 설정
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)  # Gaussian 모델 생성
    scene = Scene(dataset, gaussians)  # 씬 로드 (카메라, 이미지 등)
    gaussians.training_setup(opt)  # 옵티마이저 초기화
    
    # 체크포인트에서 재시작
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # 배경 색상 설정 (흰색 또는 검은색)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 타이밍 측정용 이벤트
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    # Depth regularization 가중치 (반복에 따라 변화)
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    # 학습용 카메라 뷰포인트 준비
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0  # 지수 이동 평균 손실
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    # === 메인 학습 루프 ===
    for iteration in range(first_iter, opt.iterations + 1):
        # 실시간 뷰어 연결 시도
        if network_gui.conn == None:
            network_gui.try_connect()
        # 뷰어와 통신 (사용자가 학습 중 실시간으로 결과 확인 가능)
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        # 학습률 업데이트 (스케줄러)
        gaussians.update_learning_rate(iteration)

        # 1000번 반복마다 Spherical Harmonics 차수를 올림 (색상 표현력 향상)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # 랜덤하게 카메라 뷰 선택 (매 반복마다 다른 각도에서 학습)
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # === 렌더링 단계 ===
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # 배경: 랜덤 색상 (데이터 증강) 또는 고정 색상
        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # Gaussian Splatting으로 이미지 렌더링
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # 알파 마스크 적용 (있는 경우)
        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # === 손실 함수 계산 ===
        gt_image = viewpoint_cam.original_image.cuda()  # Ground Truth 이미지
        Ll1 = l1_loss(image, gt_image)  # L1 손실 (픽셀 차이)
        
        # SSIM 손실 (구조적 유사도 - 인간의 시각적 인지와 비슷)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        # 전체 손실 = L1 손실 + SSIM 손실의 가중 평균
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization (깊이 정규화 - 기하학적 정확도 향상)
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        # 역전파 (그래디언트 계산)
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # === 진행률 및 로깅 ===
            # 지수 이동 평균으로 부드러운 손실 값 표시
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            # 10번 반복마다 진행률 업데이트
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # 테스트 및 저장
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # === Densification (밀도화) ===
            # Gaussian 포인트가 부족한 영역에 새로운 포인트 추가
            # 불필요한 포인트는 제거 (pruning)
            if iteration < opt.densify_until_iter:
                # 각 Gaussian의 이미지 공간 반경 추적
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                # 주기적으로 densification 수행
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                # 주기적으로 불투명도 초기화 (학습 안정화)
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # === 옵티마이저 스텝 (파라미터 업데이트) ===
            if iteration < opt.iterations:
                # Exposure 파라미터 업데이트
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                # Gaussian 파라미터 업데이트
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            # 체크포인트 저장
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):
    """출력 폴더 생성 및 TensorBoard 로거 준비"""
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # 출력 폴더 생성
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    # 설정 저장 (나중에 재현 가능하도록)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # TensorBoard writer 생성 (학습 과정 시각화)
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    """학습 과정 리포트 및 검증"""
    # TensorBoard에 손실 값 기록
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # 테스트 세트 및 학습 세트 샘플에 대한 평가
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # === 명령줄 인자 파서 설정 ===
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)  # 모델 파라미터 (데이터 경로, 해상도 등)
    op = OptimizationParams(parser)  # 최적화 파라미터 (학습률, 반복 횟수 등)
    pp = PipelineParams(parser)  # 파이프라인 파라미터 (렌더링 설정)
    parser.add_argument('--ip', type=str, default="127.0.0.1")  # 뷰어 IP
    parser.add_argument('--port', type=int, default=6009)  # 뷰어 포트
    parser.add_argument('--debug_from', type=int, default=-1)  # 디버그 시작 반복
    parser.add_argument('--detect_anomaly', action='store_true', default=False)  # 이상 감지
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])  # 테스트 시점
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])  # 저장 시점
    parser.add_argument("--quiet", action="store_true")  # 조용한 모드
    parser.add_argument('--disable_viewer', action='store_true', default=False)  # 뷰어 비활성화
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])  # 체크포인트 저장 시점
    parser.add_argument("--start_checkpoint", type=str, default = None)  # 재시작할 체크포인트
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # 시스템 상태 초기화 (난수 생성기 등)
    safe_state(args.quiet)

    # GUI 서버 시작 및 학습 실행
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)  # 그래디언트 이상 감지
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
