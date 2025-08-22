# 실행: 제공하신 경로에서 최대 4장의 DepthMap BMP를 찾아 FPFH+RANSAC 전역 정합 후 ICP로 정밀 정합하고 Poisson으로 메쉬 저장합니다.
# 결과 파일은 /mnt/data/person_result.ply 로 저장됩니다.
# 주의: 실행환경에 따라 open3d, cv2가 없을 수 있습니다.
import os, sys, glob, math, traceback
import numpy as np
from PIL import Image

try:
    import cv2
    import open3d as o3d
except Exception as e:
    raise RuntimeError("필수 라이브러리(open3d, cv2)가 설치되어 있지 않거나 로드 실패했습니다: " + str(e))

ROOT = os.path.dirname(os.path.abspath(__file__))
# 모든 가능한 경로에서 DepthMap 파일을 찾습니다
candidates = []

# 먼저 test 디렉토리에서 검색
test_dirs = ["정상", "비정상", "모름"]
view_dirs = ["오른쪽_남", "왼쪽_남", "정면_남", "후면_남"]

# test 디렉토리에서 검색
for test_dir in test_dirs:
    for view_dir in view_dirs:
        search_path = os.path.join(ROOT, "test", test_dir, view_dir, "DepthMap*.bmp")
        found = glob.glob(search_path)
        if found:
            candidates.extend(found)

# data/DepthMap 디렉토리에서도 검색
for test_dir in test_dirs:
    for view_dir in view_dirs:
        search_path = os.path.join(ROOT, "data", "DepthMap", test_dir, view_dir, "DepthMap*.bmp")
        found = glob.glob(search_path)
        if found:
            candidates.extend(found)

print("\n모든 검색 경로:")
for test_dir in test_dirs:
    for view_dir in view_dirs:
        print(" - " + os.path.join(ROOT, "test", test_dir, view_dir))
        print(" - " + os.path.join(ROOT, "data", "DepthMap", test_dir, view_dir))

print("\n검색된 모든 DepthMap 파일들:")
for c in candidates:
    print(" -", c)

# 검색된 파일들 중에서 각 방향별로 하나씩 선택
user_specified = []
used_views = set()

# 파일들을 정렬하여 일관된 순서 보장
candidates.sort()

for path in candidates:
    for view in view_dirs:
        if view in path and view not in used_views:
            if os.path.exists(path):  # 파일 존재 여부 한 번 더 확인
                print(f"Found {view} in: {path}")
                user_specified.append(path)
                used_views.add(view)
                break
# 합치고 유니크
paths = []
for p in user_specified + candidates:
    if p not in paths and os.path.exists(p):
        paths.append(p)
# 최대 4개만 사용
paths = paths[:4]

if len(paths) < 2:
    raise RuntimeError(f"찾은 뎁스맵이 부족합니다. 검색된 파일: {paths}")

print("사용할 뎁스맵 파일들:")
for p in paths:
    print(" -", p)

def intrinsics_from_fov(w,h,fov_deg=70.0):
    fov = math.radians(fov_deg)
    fx = 0.5 * w / math.tan(0.5 * fov)
    fy = fx
    cx = (w-1)/2.0
    cy = (h-1)/2.0
    return fx,fy,cx,cy

def depth_to_points(depth, depth_scale, max_depth, fx,fy,cx,cy):
    if depth.dtype != np.float32 and depth.dtype != np.float64:
        depth_m = depth.astype(np.float32) * depth_scale
    else:
        depth_m = depth.astype(np.float32)
    mask = depth_m>0
    if max_depth>0:
        mask &= (depth_m<=max_depth)
    ys, xs = np.where(mask)
    z = depth_m[ys,xs]
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    pts = np.stack([x,y,z], axis=1)
    return pts

def load_depth_points(path, fov=70.0, depth_scale=0.001, max_depth=4.0, voxel=0.004):
    print(f"파일 로딩 시도: {path}")
    if not os.path.exists(path):
        raise RuntimeError(f"파일이 존재하지 않습니다: {path}")
    
    # 경로를 UTF-8로 처리
    try:
        with open(path, 'rb') as file:
            img = Image.open(file)
            im = np.array(img)
            img.close()
    except Exception as e:
        print(f"PIL 로딩 실패, cv2로 시도합니다... 오류: {str(e)}")
        try:
            im = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if im is None:
                raise RuntimeError("cv2 이미지 로딩 실패")
        except Exception as e2:
            raise RuntimeError(f"이미지 읽기 실패: {path}, 오류: {str(e2)}")
    
    if im is None:
        raise RuntimeError("이미지 데이터가 없습니다: " + path)
    
    if im.ndim==3:
        im = np.mean(im, axis=2).astype(np.uint8)
    h,w = im.shape[:2]
    fx,fy,cx,cy = intrinsics_from_fov(w,h,fov)
    pts = depth_to_points(im, depth_scale, max_depth, fx,fy,cx,cy)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    if voxel>0:
        pcd = pcd.voxel_down_sample(voxel)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*3.0, max_nn=30))
    return pcd

# 폴더명으로 방향 추정
def infer_direction_from_path(p):
    base = os.path.basename(os.path.dirname(p))
    s = base.lower()
    if any(x in s for x in ["front","정면","앞"]): return "front"
    if any(x in s for x in ["left","왼쪽","좌"]): return "left"
    if any(x in s for x in ["right","오른쪽","우"]): return "right"
    if any(x in s for x in ["back","후면","뒤"]): return "back"
    return "unknown"

yaw_map = {"front":0.0, "left":90.0, "right":-90.0, "back":180.0, "unknown":0.0}

# 파라미터
FOV = 70.0
DEPTH_SCALE = 0.001
MAX_DEPTH = 4.0
VOXEL = 0.004

pcds = []
yaws = []
for p in paths:
    pcd = load_depth_points(p, fov=FOV, depth_scale=DEPTH_SCALE, max_depth=MAX_DEPTH, voxel=VOXEL)
    dirc = infer_direction_from_path(p)
    yaw = yaw_map.get(dirc, 0.0)
    R = pcd.get_rotation_matrix_from_axis_angle([0.0, math.radians(yaw), 0.0])
    pcd.rotate(R, center=(0,0,0))
    pcds.append(pcd)
    yaws.append((p, dirc, yaw, len(pcd.points)))

print("\n로딩 결과 (파일, 방향, yaw, 점 개수):")
for info in yaws:
    print(info)

# 기준은 첫 번째 pcd
base = pcds[0]
transforms = [np.eye(4)]
def numpy_from_pcd(pcd, max_pts=6000):
    pts = np.asarray(pcd.points)
    if len(pts) > max_pts:
        idx = np.random.choice(len(pts), max_pts, replace=False)
        pts = pts[idx]
    return pts

def compute_fpfh(pcd, voxel):
    pcd_down = pcd.voxel_down_sample(voxel)
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*2.0, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*5.0, max_nn=100)
    )
    return pcd_down, fpfh

for i in range(1, len(pcds)):
    src = pcds[i]
    # 전역 정합: FPFH + RANSAC
    src_down, fsrc = compute_fpfh(src, VOXEL)
    dst_down, fdst = compute_fpfh(base, VOXEL)
    distance_threshold = VOXEL * 1.5
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, dst_down, fsrc, fdst, True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(400000, 1000)
    )
    T_init = result_ransac.transformation
    # 정밀 정합: point-to-plane ICP
    src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL*3.0, max_nn=50))
    base.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL*3.0, max_nn=50))
    reg_icp = o3d.pipelines.registration.registration_icp(
        src, base, VOXEL*1.2, T_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80)
    )
    T_refined = reg_icp.transformation
    transforms.append(T_refined)
    print(f"\n뷰 {i}: RANSAC inlier_cnt={len(result_ransac.correspondence_set)} ; ICP fitness={reg_icp.fitness:.4f}, rmse={reg_icp.inlier_rmse:.6f}")

# 병합 및 색상 지정
colors = [
    [1, 0, 0],  # 빨강 (정면)
    [0, 1, 0],  # 초록 (우측)
    [0, 0, 1],  # 파랑 (좌측)
    [1, 1, 0]   # 노랑 (후면)
]

merged = o3d.geometry.PointCloud()
for pcd, T, color in zip(pcds, transforms, colors):
    pcd_trans = pcd.transform(T.copy())
    pcd_trans.paint_uniform_color(color)  # 각 뷰마다 다른 색상 지정
    merged += pcd_trans

# 다운샘플링
merged = merged.voxel_down_sample(VOXEL)
merged.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL*3.0, max_nn=60))

print("\n통합 포인트클라우드 점 개수:", len(merged.points))

# 결과 시각화 함수
def visualize_point_cloud(geometries, window_name="3D Point Cloud Visualization"):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1024, height=768)
    
    # 모든 지오메트리 추가
    for geometry in geometries:
        vis.add_geometry(geometry)
    
    # 렌더링 옵션 설정
    opt = vis.get_render_option()
    opt.point_size = 5.0
    opt.background_color = np.asarray([0.1, 0.1, 0.1])
    opt.show_coordinate_frame = True
    opt.light_on = True
    
    # 카메라 설정
    ctr = vis.get_view_control()
    ctr.set_zoom(0.5)
    ctr.set_lookat([0, 0, 0])
    ctr.set_front([1.0, 0.5, 2.0])
    ctr.set_up([0.0, -1.0, 0.0])
    
    # 초기 업데이트
    vis.update_geometry(merged)
    vis.poll_events()
    vis.update_renderer()
    
    print(f"\n{window_name}이 열립니다. 종료하려면 창을 닫으세요.")
    print("마우스 조작:")
    print(" - 회전: 좌클릭 드래그")
    print(" - 이동: 우클릭 드래그")
    print(" - 확대/축소: 마우스 휠")
    
    # 뷰어 실행
    vis.run()
    vis.destroy_window()

# 좌표계 생성
coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100)

print("\n시각화 시작...")
print("포인트 클라우드 점 개수:", len(merged.points))

# 먼저 개별 포인트 클라우드 시각화
for i, pcd in enumerate(pcds):
    print(f"\n뷰 {i} 시각화...")
    pcd.paint_uniform_color(colors[i])
    visualize_point_cloud([pcd, coord_frame], f"View {i} - Points: {len(pcd.points)}")

# 최종 병합된 포인트 클라우드 시각화
print("\n최종 병합 결과 시각화...")
visualize_point_cloud([merged, coord_frame], "Final Merged Point Cloud")

