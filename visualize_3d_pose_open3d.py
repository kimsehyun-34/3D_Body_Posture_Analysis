import numpy as np
import cv2
import open3d as o3d

def load_depth_map(file_path):
    # PIL을 사용하여 이미지 로드
    from PIL import Image
    try:
        with Image.open(file_path) as img:
            depth_map = np.array(img)
            if len(depth_map.shape) > 2:  # Convert RGB to grayscale if needed
                depth_map = np.mean(depth_map, axis=2).astype(np.uint8)
            
            # 정사각형으로 자르기
            height, width = depth_map.shape
            size = min(height, width)
            
            # 중앙 기준으로 자르기
            start_y = (height - size) // 2
            start_x = (width - size) // 2
            depth_map = depth_map[start_y:start_y+size, start_x:start_x+size]
            
            return depth_map.astype(np.float32) / 255.0  # Normalize to [0,1]
    except Exception as e:
        print(f"Failed to load: {file_path}")
        print(f"Error: {str(e)}")
        return None

def create_point_cloud_from_depth(depth_map, view):
    if depth_map is None:
        return None
        
    size = depth_map.shape[0]  # 정사각형이므로 한 변의 길이만 필요
    y, x = np.mgrid[0:size, 0:size]
    
    # 포인트 수를 줄이기 위해 다운샘플링
    step = 1
    x = x[::step, ::step]
    y = y[::step, ::step]
    depth_map = depth_map[::step, ::step]
    
    # 중심점 조정을 위한 오프셋 계산
    x = x - size/2
    y = y - size/2
    
    scale = 100  # 스케일 조정
    
    # 뷰에 따라 좌표 변환
    if view == "front":
        points = np.stack([x, -y, depth_map * scale], axis=-1)
    elif view == "right":
        points = np.stack([depth_map * scale * 3, -y, -x], axis=-1)  # 우측 깊이 2배
    elif view == "left":
        points = np.stack([-depth_map * scale * 3, -y, x], axis=-1)  # 좌측 깊이 2배
    elif view == "back":
        points = np.stack([-x, -y, -depth_map * scale], axis=-1)

    # 유효한 깊이값을 가진 포인트만 선택 (임계값 0.3 적용)
    threshold = 0.2  # 30% 이상의 깊이값만 사용
    valid_points = points[depth_map > threshold]
    
    # 너무 많은 포인트가 있는 경우 추가 다운샘플링
    if len(valid_points) > 15000:
        indices = np.random.choice(len(valid_points), 15000, replace=False)
        valid_points = valid_points[indices]
    
    # Open3D 포인트 클라우드 생성
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points)
    
    # 뷰에 따른 색상 지정
    colors = {
        "front": [1, 0, 0],  # 빨간색
        "right": [0, 1, 0],  # 초록색
        "left": [0, 0, 1],   # 파란색
        "back": [1, 1, 0]    # 노란색
    }
    pcd.paint_uniform_color(colors[view])
    
    return pcd

def align_point_clouds(source, target, threshold=10):
    # 초기 변환 행렬 (4x4 단위 행렬)
    init_transformation = np.eye(4)
    
    # ICP 정렬 수행
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance=threshold,
        init=init_transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6,
            relative_rmse=1e-6,
            max_iteration=50
        )
    )
    
    # 결과가 유효한 경우에만 변환 적용
    if reg_p2p.fitness > 0.01:  # 정렬 품질이 1% 이상인 경우
        return source.transform(reg_p2p.transformation)
    return source  # 정렬이 실패한 경우 원본 반환

def visualize_3d_pose():
    # 각 뷰의 DepthMap 로드
    views = {
        "front": r"d:\기타\파일 자료\파일\프로젝트 PJ\AAAAAA2_3D_자세\test\정상\정면_남\DepthMap0.bmp",
        "right": r"d:\기타\파일 자료\파일\프로젝트 PJ\AAAAAA2_3D_자세\test\정상\오른쪽_남\DepthMap0.bmp",
        "left": r"d:\기타\파일 자료\파일\프로젝트 PJ\AAAAAA2_3D_자세\test\정상\왼쪽_남\DepthMap0.bmp",
        "back": r"d:\기타\파일 자료\파일\프로젝트 PJ\AAAAAA2_3D_자세\test\정상\후면_남\DepthMap0.bmp"
    }
    
    # 각 뷰의 포인트 클라우드 생성
    point_clouds = {}
    for view_name, file_path in views.items():
        depth_map = load_depth_map(file_path)
        if depth_map is not None:
            pcd = create_point_cloud_from_depth(depth_map, view_name)
            if pcd is not None:
                # 법선 벡터 계산
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
                point_clouds[view_name] = pcd
    
    # 정면과 후면을 먼저 정렬
    aligned_clouds = [point_clouds["front"]]
    if "back" in point_clouds:
        back_aligned = align_point_clouds(point_clouds["back"], point_clouds["front"], threshold=50)
        aligned_clouds.append(back_aligned)
    
    # 정렬된 정면을 기준으로 좌우 정렬
    target = point_clouds["front"]
    for view in ["right", "left"]:
        if view in point_clouds:
            # 좌우는 더 큰 threshold 값을 사용
            aligned = align_point_clouds(point_clouds[view], target, threshold=100)
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
    
    # 좌표축 생성
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100)
    
    # 초기 카메라 뷰포인트 설정
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Pose Visualization", width=1024, height=768)
    
    # 병합된 포인트 클라우드와 좌표축 추가
    vis.add_geometry(merged_cloud)
    vis.add_geometry(coord_frame)
    
    # 렌더링 옵션 설정
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.asarray([0, 0, 0])  # 검은색 배경
    
    # 카메라 위치 설정
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    ctr.set_front([0.5, -0.5, -0.5])
    ctr.set_up([0, -1, 0])
    
    # 시각화
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    visualize_3d_pose()
