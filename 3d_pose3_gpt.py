#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DepthMap 멀티뷰 3D 재구성 스크립트

데이터 구조 예시
 data/DepthMap/
 ├── [방향]_[성별](자세상태)/
 │   ├── DepthMap1.bmp
 │   ├── DepthMap2.bmp
 │   └── ...

- 같은 사람의 왼쪽/오른쪽/정면/후면 뎁스맵을 통합하여 3D 포인트클라우드 및 메쉬를 생성합니다.
- 카메라 내·외부 파라미터가 없는 상황을 가정하여, FOV 기반의 단순화된 내참수와 방향별 Yaw 회전으로 정렬합니다.
- 필요 시 파라미터를 조정해 밀도/스케일을 보정하십시오.

필수 패키지: open3d, opencv-python, numpy

사용 방법(예):
  python reconstruct_from_depthmaps.py \
    --root data/DepthMap \
    --person "front_M(normal)" "left_M(normal)" "right_M(normal)" "back_M(normal)" \
    --fov 70 \
    --depth-scale 0.001 \
    --max-depth 4.0 \
    --voxel 0.005 \
    --to-mesh poisson

참고:
- depth-scale: BMP 픽셀값 × depth-scale = 미터 단위 깊이. (예: 밀리미터 저장이면 0.001)
- max-depth: 최대 사용 깊이(미터).
- fov: 수평 시야각(도). 장비에 맞게 조정.
- person 인자에 동일 인물의 각 방향 폴더명을 넣으십시오(영문/한글 방향명 모두 허용).
"""

import os
import re
import glob
import math
import argparse
from typing import Dict, List, Tuple

import cv2
import numpy as np
import open3d as o3d

# -----------------------------
# 유틸: 방향명 파싱 및 Yaw 각도 매핑
# -----------------------------
DIRECTION_ALIASES = {
    "front": ["front", "정면", "앞"],
    "back": ["back", "후면", "뒤"],
    "left": ["left", "왼쪽", "좌"],
    "right": ["right", "오른쪽", "우"],
}

YAW_BY_DIRECTION_DEG = {
    # 카메라 좌표계: +Z 정면, +X 오른쪽, +Y 아래(이미지 기준),
    # 인물 좌표계에서 정면이 0°, 좌/우가 ±90°, 후면이 180°로 가정
    "front": 0.0,
    "left": +90.0,
    "right": -90.0,
    "back": 180.0,
}


def infer_direction_from_folder(folder_name: str) -> str:
    base = os.path.basename(folder_name)
    key = base.split("_")[0]  # "[방향]_[성별](자세)"에서 앞부분 취득
    for canonical, aliases in DIRECTION_ALIASES.items():
        for a in aliases:
            if a.lower() in key.lower():
                return canonical
    # 못 찾으면 정면으로 fallback하되 로그 남김
    print(f"[WARN] 방향 추론 실패: '{folder_name}'. 'front'로 처리합니다.")
    return "front"


# -----------------------------
# 카메라 내참수 구성(FOV 기반) & 깊이→3D 변환
# -----------------------------

def intrinsics_from_fov(width: int, height: int, fov_deg: float) -> Tuple[float, float, float, float]:
    fov_rad = math.radians(fov_deg)
    fx = 0.5 * width / math.tan(0.5 * fov_rad)
    fy = fx  # 정방 가정
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    return fx, fy, cx, cy


def depth_to_points(depth: np.ndarray, depth_scale: float, max_depth_m: float, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Depth(HxW, uint8/uint16/float) -> (N,3) XYZ in meters, Z-forward.
    """
    if depth.dtype != np.float32 and depth.dtype != np.float64:
        depth_m = depth.astype(np.float32) * depth_scale
    else:
        depth_m = depth.astype(np.float32)

    if max_depth_m > 0:
        mask = (depth_m > 0) & (depth_m <= max_depth_m)
    else:
        mask = depth_m > 0

    ys, xs = np.where(mask)
    z = depth_m[ys, xs]
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy

    pts = np.stack([x, y, z], axis=1)
    return pts


# -----------------------------
# Open3D 변환/클라우드 유틸
# -----------------------------

def make_o3d_cloud(points: np.ndarray, voxel: float = 0.0) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if voxel and voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    return pcd


def rotate_cloud_yaw(pcd: o3d.geometry.PointCloud, yaw_deg: float) -> o3d.geometry.PointCloud:
    yaw = math.radians(yaw_deg)
    R = pcd.get_rotation_matrix_from_axis_angle([0, yaw, 0])
    pcd_rot = pcd.rotate(R, center=(0, 0, 0))
    return pcd_rot


def remove_outliers(pcd: o3d.geometry.PointCloud, nb_neighbors=20, std_ratio=2.0) -> o3d.geometry.PointCloud:
    cl, idx = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return cl


# -----------------------------
# 메쉬 재구성(선택: Poisson or Ball Pivoting)
# -----------------------------

def to_mesh(pcd: o3d.geometry.PointCloud, method: str = "poisson") -> o3d.geometry.TriangleMesh:
    if method == "poisson":
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9, n_threads=0
        )
        # 낮은 밀도 영역 컷오프로 홀 제거 보정
        densities = np.asarray(densities)
        density_thr = np.quantile(densities, 0.02)
        verts_to_keep = densities > density_thr
        mesh = mesh.select_by_index(np.where(verts_to_keep)[0])
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        mesh.compute_vertex_normals()
        return mesh
    elif method == "bpa":
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=50))
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = 2.5 * avg_dist
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector([radius, radius * 2])
        )
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        mesh.compute_vertex_normals()
        return mesh
    else:
        raise ValueError("Unsupported meshing method. Use 'poisson' or 'bpa'.")


# -----------------------------
# 파일 로딩
# -----------------------------

def load_depth_images_from_folder(folder: str) -> List[np.ndarray]:
    paths = sorted(glob.glob(os.path.join(folder, "*.bmp")))
    imgs = []
    for p in paths:
        im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if im is None:
            print(f"[WARN] 이미지 읽기 실패: {p}")
            continue
        if len(im.shape) == 3:
            # 3채널일 경우 첫 채널 사용 또는 그레이 변환
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        imgs.append(im)
    return imgs


def build_cloud_from_folder(folder: str, fov_deg: float, depth_scale: float, max_depth_m: float, voxel: float) -> o3d.geometry.PointCloud:
    depth_list = load_depth_images_from_folder(folder)
    if not depth_list:
        raise RuntimeError(f"폴더에 BMP가 없습니다: {folder}")

    # 여러 장이 있으면 중앙값으로 노이즈 완화(간단한 융합)
    stack = np.stack(depth_list, axis=0).astype(np.float32)
    median_depth = np.median(stack, axis=0).astype(depth_list[0].dtype)

    h, w = median_depth.shape
    fx, fy, cx, cy = intrinsics_from_fov(w, h, fov_deg)
    pts = depth_to_points(median_depth, depth_scale, max_depth_m, fx, fy, cx, cy)
    pcd = make_o3d_cloud(pts, voxel=voxel)
    return pcd


# -----------------------------
# 시각화 및 파이프라인
# -----------------------------

def reconstruct_person(root: str, person_dirs: List[str], fov_deg: float, depth_scale: float, max_depth_m: float,
                        voxel: float, outlier_nb: int, outlier_std: float, meshing: str,
                        save_mesh: str = None) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh]:
    clouds = []
    for d in person_dirs:
        full = os.path.join(root, d)
        if not os.path.isdir(full):
            raise RuntimeError(f"폴더가 존재하지 않습니다: {full}")
        direction = infer_direction_from_folder(d)
        yaw = YAW_BY_DIRECTION_DEG[direction]
        print(f"[INFO] 폴더='{d}' → 방향='{direction}' → Yaw={yaw}°")
        pcd = build_cloud_from_folder(full, fov_deg, depth_scale, max_depth_m, voxel)
        pcd = rotate_cloud_yaw(pcd, yaw)
        clouds.append(pcd)

    # 통합
    merged = o3d.geometry.PointCloud()
    for c in clouds:
        merged += c

    # 아웃라이어 제거 및 재표본화
    merged = remove_outliers(merged, nb_neighbors=outlier_nb, std_ratio=outlier_std)
    if voxel and voxel > 0:
        merged = merged.voxel_down_sample(voxel)
    merged.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=50))

    mesh = to_mesh(merged, method=meshing) if meshing else None

    if save_mesh and mesh is not None:
        ext = os.path.splitext(save_mesh)[1].lower()
        if ext not in [".ply", ".stl", ".obj", ".glb", ".gltf"]:
            print("[WARN] 지원 확장자(.ply/.stl/.obj/.glb/.gltf)가 아닙니다. .ply로 저장합니다.")
            save_mesh = os.path.splitext(save_mesh)[0] + ".ply"
        o3d.io.write_triangle_mesh(save_mesh, mesh)
        print(f"[INFO] 메쉬 저장: {save_mesh}")

    # 시각화
    to_draw = [merged]
    if mesh is not None:
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.9, 0.9, 0.9])
        to_draw.append(mesh)
    o3d.visualization.draw_geometries(to_draw)

    return merged, mesh


def main():
    ap = argparse.ArgumentParser(description="멀티뷰 DepthMap 3D 재구성")
    ap.add_argument("--root", type=str, default="data/DepthMap", help="루트 폴더")
    ap.add_argument("--person", type=str, nargs='+', required=True, help="동일 인물의 방향별 폴더명 나열")
    ap.add_argument("--fov", type=float, default=70.0, help="수평 FOV(도)")
    ap.add_argument("--depth-scale", type=float, default=0.001, help="픽셀값×scale=미터")
    ap.add_argument("--max-depth", type=float, default=4.0, help="최대 깊이(m), 0이면 제한 없음")
    ap.add_argument("--voxel", type=float, default=0.005, help="다운샘플링 보xel 크기(m)")
    ap.add_argument("--outlier-nb", type=int, default=20, help="아웃라이어 제거 이웃 수")
    ap.add_argument("--outlier-std", type=float, default=2.0, help="아웃라이어 표준편차 임계")
    ap.add_argument("--to-mesh", type=str, default="poisson", choices=["poisson", "bpa"], help="메쉬 방법")
    ap.add_argument("--save-mesh", type=str, default=None, help="결과 메쉬 저장 경로(.ply/.stl/.obj/.glb/.gltf)")
    args = ap.parse_args()

    reconstruct_person(
        root=args.root,
        person_dirs=args.person,
        fov_deg=args.fov,
        depth_scale=args.depth_scale,
        max_depth_m=args.max_depth,
        voxel=args.voxel,
        outlier_nb=args.outlier_nb,
        outlier_std=args.outlier_std,
        meshing=args.to_mesh,
        save_mesh=args.save_mesh,
    )


if __name__ == "__main__":
    main()
