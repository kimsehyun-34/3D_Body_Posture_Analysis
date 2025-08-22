import numpy as np
import cv2
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
            
            # 가우시안 블러로 노이즈 감소
            depth_map = cv2.GaussianBlur(depth_map, (3, 3), 0)
            
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
    
    step = 1  # 포인트 수를 늘리기 위해 step 크기 감소
    x = x[::step, ::step]
    y = y[::step, ::step]
    depth_map = depth_map[::step, ::step]
    
    x = x - size/2
    y = y - size/2
    
    scale = 100
    depth_scale = scale * 1.5  # 깊이 스케일 조정
    
    if view == "front":
        points = np.stack([x, -y, depth_map * scale], axis=-1)
    elif view == "right":
        points = np.stack([depth_map * depth_scale, -y, -x], axis=-1)
    elif view == "left":
        points = np.stack([-depth_map * depth_scale, -y, x], axis=-1)
    elif view == "back":
        points = np.stack([-x, -y, -depth_map * scale], axis=-1)

    threshold = 0.2
    valid_points = points[depth_map > threshold]
    
    if len(valid_points) > 20000:  # 포인트 수 증가
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
    
    return pcd

def create_initial_transform(source_view, target_view="front"):
    transform = np.eye(4)
    
    if source_view == "left":
        # 왼쪽 뷰는 90도 회전 + Y축 이동
        angle = np.pi / 2
        transform[0:3, 0:3] = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
        transform[0, 3] = -100  # X축 방향으로 이동
        
    elif source_view == "right":
        # 오른쪽 뷰는 -90도 회전 + Y축 이동
        angle = -np.pi / 2
        transform[0:3, 0:3] = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
        transform[0, 3] = 100  # X축 방향으로 이동
        
    elif source_view == "back":
        # 후면 뷰는 180도 회전
        transform[0:3, 0:3] = np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, -1]
        ])
    
    return transform

def align_point_clouds(source, target, source_view="", target_view="front", threshold=10):
    # 초기 변환 적용
    init_transform = create_initial_transform(source_view, target_view)
    source_transformed = source.transform(init_transform)
    
    # 법선 벡터 재계산
    source_transformed.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))
    target.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))
    
    # Point-to-plane ICP 실행
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_transformed, target,
        max_correspondence_distance=threshold,
        init=np.eye(4),  # 이미 초기 변환을 적용했으므로 단위 행렬 사용
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6,
            relative_rmse=1e-6,
            max_iteration=100
        )
    )
    
    if reg_p2p.fitness > 0.01:
        # 최종 변환은 초기 변환과 ICP 변환의 조합
        final_transform = np.dot(reg_p2p.transformation, init_transform)
        return source.transform(final_transform)
    return source_transformed  # ICP가 실패하면 초기 변환된 상태 반환

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
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
                point_clouds[view_name] = pcd
    
    # 정면을 기준으로 정렬
    aligned_clouds = [point_clouds["front"]]
    target = point_clouds["front"]
    
    # 정렬 순서: 후면 -> 좌우
    alignment_order = [
        ("back", 50),    # 후면은 작은 threshold
        ("right", 100),  # 좌우는 큰 threshold
        ("left", 100)
    ]
    
    for view, threshold in alignment_order:
        if view in point_clouds:
            print(f"Aligning {view} view...")
            aligned = align_point_clouds(
                point_clouds[view],
                target,
                source_view=view,
                target_view="front",
                threshold=threshold
            )
            aligned_clouds.append(aligned)
    
    # 모든 포인트 클라우드를 하나로 합치기
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
    
    # 좌표축 생성
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100)
    
    # 시각화
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
