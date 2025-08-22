import numpy as np
import cv2
import open3d as o3d

def load_depth_map(file_path):
    from PIL import Image
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
            
            # 가우시안 블러로 노이즈 감소
            depth_map = cv2.GaussianBlur(depth_map, (3, 3), 0)
            
            return depth_map.astype(np.float32) / 255.0
    except Exception as e:
        print(f"Failed to load: {file_path}")
        print(f"Error: {str(e)}")
        return None

def remove_outliers(pcd):
    # 통계적 이상치 제거
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.select_by_index(ind)
    
    # 반경 기반 이상치 제거
    cl, ind = pcd.remove_radius_outlier(nb_points=16, radius=10)
    pcd = pcd.select_by_index(ind)
    
    return pcd

def create_point_cloud_from_depth(depth_map, view):
    if depth_map is None:
        return None
        
    size = depth_map.shape[0]
    y, x = np.mgrid[0:size, 0:size]
    
    step = 1
    x = x[::step, ::step]
    y = y[::step, ::step]
    depth_map = depth_map[::step, ::step]
    
    x = x - size/2
    y = y - size/2
    
    scale = 100
    
    if view == "front":
        points = np.stack([x, -y, depth_map * scale], axis=-1)
    elif view == "right":
        points = np.stack([depth_map * scale * 3, -y, -x], axis=-1)
    elif view == "left":
        points = np.stack([-depth_map * scale * 3, -y, x], axis=-1)
    elif view == "back":
        points = np.stack([-x, -y, -depth_map * scale], axis=-1)

    # 깊이 임계값 증가 및 엣지 제거
    threshold = 0.35
    edge_threshold = 50  # 엣지에서의 깊이 차이 임계값
    
    # 엣지 검출을 위한 깊이 차이 계산
    depth_diff_x = np.abs(np.diff(depth_map, axis=1))
    depth_diff_y = np.abs(np.diff(depth_map, axis=0))
    
    # 패딩을 추가하여 원본 크기와 맞추기
    depth_diff_x = np.pad(depth_diff_x, ((0, 0), (0, 1)), mode='edge')
    depth_diff_y = np.pad(depth_diff_y, ((0, 1), (0, 0)), mode='edge')
    
    # 엣지가 아닌 지점 선택
    valid_edges = (depth_diff_x < edge_threshold/scale) & (depth_diff_y < edge_threshold/scale)
    valid_depth = depth_map > threshold
    
    valid_points = points[valid_edges & valid_depth]
    
    if len(valid_points) > 20000:
        indices = np.random.choice(len(valid_points), 20000, replace=False)
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
    
    # 법선 벡터 계산 및 이상치 제거
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
    pcd = remove_outliers(pcd)
    
    return pcd

def align_point_clouds(source, target, threshold=10):
    init_transformation = np.eye(4)
    
    # 포인트 클라우드 다운샘플링
    source_down = source.voxel_down_sample(voxel_size=2.0)
    target_down = target.voxel_down_sample(voxel_size=2.0)
    
    # 법선 벡터 재계산
    source_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))
    target_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))
    
    # Point-to-plane ICP 사용
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_down, target_down,
        max_correspondence_distance=threshold,
        init=init_transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6,
            relative_rmse=1e-6,
            max_iteration=50
        )
    )
    
    if reg_p2p.fitness > 0.03:
        return source.transform(reg_p2p.transformation)
    return source

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
    
    aligned_clouds = [point_clouds["front"]]
    if "back" in point_clouds:
        back_aligned = align_point_clouds(point_clouds["back"], point_clouds["front"], threshold=50)
        aligned_clouds.append(back_aligned)
    
    target = point_clouds["front"]
    for view in ["right", "left"]:
        if view in point_clouds:
            aligned = align_point_clouds(point_clouds[view], target, threshold=100)
            aligned_clouds.append(aligned)
    
    # 모든 포인트 클라우드를 하나로 합치고 최종 노이즈 제거
    merged_cloud = o3d.geometry.PointCloud()
    points = []
    colors = []
    for pcd in aligned_clouds:
        points.extend(np.asarray(pcd.points))
        colors.extend(np.asarray(pcd.colors))
    merged_cloud.points = o3d.utility.Vector3dVector(np.array(points))
    merged_cloud.colors = o3d.utility.Vector3dVector(np.array(colors))
    
    # 최종 노이즈 제거 및 다운샘플링
    merged_cloud = merged_cloud.voxel_down_sample(voxel_size=2.0)
    merged_cloud = remove_outliers(merged_cloud)
    
    # 좌표축 생성
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Pose Visualization (Denoised)", width=1024, height=768)
    
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
