import numpy as np
import open3d as o3d
from PIL import Image

def load_depth_map(file_path):
    try:
        with Image.open(file_path) as img:
            depth_map = np.array(img)
            if len(depth_map.shape) > 2:
                depth_map = np.mean(depth_map, axis=2).astype(np.uint8)
            
            height, width = depth_map.shape
            size = min(height, width)
            
            start_y = (height - size) // 2
            start_x = (width - size) // 2
            depth_map = depth_map[start_y:start_y+size, start_x:start_x+size]
            
            return depth_map.astype(np.float32) / 255.0
    except Exception as e:
        print(f"Failed to load: {file_path}")
        print(f"Error: {str(e)}")
        return None

def create_point_cloud_from_depth(depth_map, view):
    if depth_map is None:
        return None
        
    size = depth_map.shape[0]
    y, x = np.mgrid[0:size, 0:size]
    
    step = 2
    x = x[::step, ::step]
    y = y[::step, ::step]
    depth_map = depth_map[::step, ::step]
    
    x = x - size/2
    y = y - size/2
    
    scale = 100
    
    if view == "front":
        points = np.stack([x, -y, depth_map * scale], axis=-1)
    elif view == "right":
        points = np.stack([depth_map * scale * 2, -y, -x], axis=-1)
    elif view == "left":
        points = np.stack([-depth_map * scale * 2, -y, x], axis=-1)
    elif view == "back":
        points = np.stack([-x, -y, -depth_map * scale], axis=-1)

    threshold = 0.2
    valid_points = points[depth_map > threshold]
    
    if len(valid_points) > 15000:
        indices = np.random.choice(len(valid_points), 15000, replace=False)
        valid_points = valid_points[indices]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points)
    
    colors = {
        "front": [1, 0, 0],
        "right": [0, 1, 0],
        "left": [0, 0, 1],
        "back": [1, 1, 0]
    }
    pcd.paint_uniform_color(colors[view])
    
    return pcd

def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    
    return pcd_down, pcd_fpfh

def prepare_dataset(source, target, voxel_size):
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )
    return result

def refine_registration(source, target, result_ransac, voxel_size):
    distance_threshold = voxel_size * 0.4
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return result

def create_initial_transform(view):
    transform = np.eye(4)
    if view == "left":
        transform[0:3, 0:3] = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0]
        ])
    elif view == "right":
        transform[0:3, 0:3] = np.array([
            [0, 0, -1],
            [0, 1, 0],
            [1, 0, 0]
        ])
    elif view == "back":
        transform[0:3, 0:3] = np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, -1]
        ])
    return transform

def align_point_clouds(source, target, view, voxel_size=3.0):
    # 초기 변환 적용
    init_transform = create_initial_transform(view)
    source_transformed = source.transform(init_transform)
    
    # 전처리 및 특징 추출
    source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(
        source_transformed, target, voxel_size)
    
    # 전역 정합
    result_ransac = execute_global_registration(
        source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    
    # 지역 정합으로 미세 조정
    result_icp = refine_registration(
        source_transformed, target, result_ransac, voxel_size)
    
    return source.transform(np.dot(result_icp.transformation, init_transform))

def visualize_3d_pose():
    views = {
        "front": r"d:\기타\파일 자료\파일\프로젝트 PJ\AAAAAA2_3D_자세\test\정상\정면_남\DepthMap0.bmp",
        "right": r"d:\기타\파일 자료\파일\프로젝트 PJ\AAAAAA2_3D_자세\test\정상\오른쪽_남\DepthMap0.bmp",
        "left": r"d:\기타\파일 자료\파일\프로젝트 PJ\AAAAAA2_3D_자세\test\정상\왼쪽_남\DepthMap0.bmp",
        "back": r"d:\기타\파일 자료\파일\프로젝트 PJ\AAAAAA2_3D_자세\test\정상\후면_남\DepthMap0.bmp"
    }
    
    point_clouds = {}
    for view_name, file_path in views.items():
        depth_map = load_depth_map(file_path)
        if depth_map is not None:
            pcd = create_point_cloud_from_depth(depth_map, view_name)
            if pcd is not None:
                point_clouds[view_name] = pcd
    
    # 정면을 기준으로 정렬
    aligned_clouds = [point_clouds["front"]]
    target = point_clouds["front"]
    
    # 정렬 순서 변경: 먼저 후면, 그 다음 좌우
    alignment_order = ["back", "right", "left"]
    
    for view in alignment_order:
        if view in point_clouds:
            print(f"Aligning {view} view...")
            aligned = align_point_clouds(
                point_clouds[view],
                target,
                view,
                voxel_size=3.0
            )
            aligned_clouds.append(aligned)
    
    # 포인트 클라우드 병합
    merged_cloud = o3d.geometry.PointCloud()
    points = []
    colors = []
    for pcd in aligned_clouds:
        points.extend(np.asarray(pcd.points))
        colors.extend(np.asarray(pcd.colors))
    merged_cloud.points = o3d.utility.Vector3dVector(np.array(points))
    merged_cloud.colors = o3d.utility.Vector3dVector(np.array(colors))
    
    # 노이즈 제거 및 다운샘플링
    merged_cloud = merged_cloud.voxel_down_sample(voxel_size=2.0)
    
    # 통계적 이상치 제거
    cl, ind = merged_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    merged_cloud = merged_cloud.select_by_index(ind)
    
    # 시각화
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Pose Visualization", width=1024, height=768)
    
    vis.add_geometry(merged_cloud)
    vis.add_geometry(coord_frame)
    
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.asarray([0, 0, 0])
    
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    ctr.set_front([0.5, -0.5, -0.5])
    ctr.set_up([0, -1, 0])
    
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    visualize_3d_pose()
