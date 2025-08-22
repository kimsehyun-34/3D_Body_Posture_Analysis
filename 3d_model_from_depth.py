import open3d as o3d
import numpy as np
import cv2
import os
import copy
from pathlib import Path

def load_depth_map(file_path):
    """
    BMP 형식의 깊이 맵을 로드하고 numpy 배열로 변환합니다.
    한글 경로를 지원하기 위해 PIL을 사용합니다.
    """
    print(f"Loading depth map: {file_path}")
    # PIL을 사용하여 이미지 로드 (한글 경로 지원)
    from PIL import Image
    try:
        with Image.open(file_path) as img:
            depth_img = np.array(img)
            if len(depth_img.shape) > 2:  # RGB 이미지인 경우 그레이스케일로 변환
                depth_img = np.mean(depth_img, axis=2).astype(np.uint8)
            
            # 정사각형으로 자르기
            height, width = depth_img.shape
            size = min(height, width)
            
            # 중앙 기준으로 자르기
            start_y = (height - size) // 2
            start_x = (width - size) // 2
            depth_img = depth_img[start_y:start_y+size, start_x:start_x+size]
            
            # 8비트 이미지를 float로 변환하고 정규화
            depth_array = depth_img.astype(np.float32) / 255.0
            
            # 시각화를 위해 출력
            print(f"Depth map shape: {depth_array.shape}, min: {np.min(depth_array)}, max: {np.max(depth_array)}")
            
            return depth_array
    except Exception as e:
        print(f"Failed to load: {file_path}")
        print(f"Error: {str(e)}")
        raise ValueError(f"Could not load depth map: {file_path}")

def create_point_cloud_from_depth(depth_array, view="정면"):
    """
    깊이 맵에서 포인트 클라우드를 생성합니다.
    각 뷰(정면, 후면, 왼쪽, 오른쪽)에 대해 적절한 좌표 변환을 적용합니다.
    모든 뷰가 동일한 스케일과 크기를 갖도록 보장합니다.
    """
    if depth_array is None:
        return None
        
    size = depth_array.shape[0]  # 정사각형이므로 한 변의 길이만 필요
    y, x = np.mgrid[0:size, 0:size]
    
    # 균일한 다운샘플링 적용 (모든 뷰에 동일한 해상도)
    step = 2  # 모든 뷰에 일관된 스텝 적용
    x = x[::step, ::step]
    y = y[::step, ::step]
    depth_array = depth_array[::step, ::step]
    
    # 깊이 맵 노이즈 감소 (가우시안 블러 적용)
    kernel_size = 3
    depth_array = cv2.GaussianBlur(depth_array, (kernel_size, kernel_size), 0)
    
    # 중심점 조정을 위한 오프셋 계산
    x = x - size/2
    y = y - size/2
    
    # 모든 뷰에 대해 일관된 스케일 적용
    scale = 300  # 모든 뷰에 동일한 스케일 적용
    
    # 뷰에 따라 좌표 변환 (3D 공간에서 올바르게 배치)
    # 초기 위치 지정 없이 좌표계 변환만 적용 (정렬 알고리즘이 위치를 결정)
    if view == "정면":
        # 정면 뷰: Z축이 카메라 방향
        points = np.stack([x, -y, depth_array * scale], axis=-1)
    elif view == "오른쪽":
        # 오른쪽 뷰: X축이 카메라 방향, Z축은 원래 X축의 반대 방향
        points = np.stack([depth_array * scale, -y, -x], axis=-1)
    elif view == "왼쪽":
        # 왼쪽 뷰: -X축이 카메라 방향, Z축은 원래 X축 방향
        points = np.stack([-depth_array * scale, -y, x], axis=-1)
    elif view == "후면":
        # 후면 뷰: -Z축이 카메라 방향
        points = np.stack([-x, -y, -depth_array * scale], axis=-1)
    
    # 유효한 깊이값을 가진 포인트만 선택 (임계값 적용)
    threshold = 0.25  # 모든 뷰에 동일한 임계값 적용
    valid_points = points[depth_array > threshold]
    
    # 일관된 필터링: 극단값 제거
    if len(valid_points) > 0:
        # 각 축에 대해 동일한 분위수 적용
        percentile_low = 1.0
        percentile_high = 99.0
        for axis in range(3):
            low = np.percentile(valid_points[:, axis], percentile_low)
            high = np.percentile(valid_points[:, axis], percentile_high)
            mask = (valid_points[:, axis] >= low) & (valid_points[:, axis] <= high)
            valid_points = valid_points[mask]
    
    # 모든 뷰에 대해 일관된 포인트 수 관리
    target_points = 20000  # 모든 뷰에 동일한 목표 포인트 수
    if len(valid_points) > target_points:
        indices = np.random.choice(len(valid_points), target_points, replace=False)
        valid_points = valid_points[indices]
    elif len(valid_points) < 1000:
        print(f"Warning: Too few points ({len(valid_points)}) for {view} view")
    
    # Open3D 포인트 클라우드 생성
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points)
    
    # 디버깅을 위한 색상 설정
    colors = {
        "정면": [1, 0, 0],  # 빨간색
        "오른쪽": [0, 1, 0],  # 초록색
        "왼쪽": [0, 0, 1],   # 파란색
        "후면": [1, 1, 0]    # 노란색
    }
    
    pcd.paint_uniform_color(colors[view])
    
    # 모든 뷰에 대해 일관된 법선 벡터 계산 파라미터
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=30)
    
    print(f"Point cloud created with {len(pcd.points)} points for {view} view")
    
    return pcd

def align_point_clouds(pcds, vis=False):
    """
    ICP 알고리즘을 사용하여 여러 포인트 클라우드를 정렬합니다.
    정렬 순서: 
    1. 정면을 기준으로 배치
    2. 정면을 기준으로 좌, 우 포인트 클라우드 정렬
    3. 정렬된 좌, 우 포인트 클라우드에 후면을 정렬
    """
    if len(pcds) < 2:
        return pcds
    
    print("Starting sequential alignment process...")
    
    # 1. 전처리: 모든 포인트 클라우드 준비
    processed_pcds = []
    for i, pcd in enumerate(pcds):
        print(f"Preprocessing point cloud {i}...")
        # 복사본 생성
        processed_pcd = copy.deepcopy(pcd)
        
        # 이상치 제거
        cl, ind = processed_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        processed_pcd = cl
        print(f"  Removed outliers from point cloud {i}, {len(processed_pcd.points)} points remaining")
        
        # 볼셀 다운샘플링 적용 (모든 뷰에 동일한 볼셀 크기)
        processed_pcd = processed_pcd.voxel_down_sample(voxel_size=3.0)
        print(f"  Downsampled point cloud {i}, now has {len(processed_pcd.points)} points")
        
        # 법선 벡터 계산 (모든 뷰에 동일한 파라미터)
        processed_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30)
        )
        
        processed_pcds.append(processed_pcd)
    
    # 2. 정렬을 위한 파라미터 설정
    distance_threshold = 30.0  # ICP 거리 임계값
    
    # 3. 정확한 초기 변환 행렬 설정 (정확한 90도/180도 회전)
    # 오른쪽 뷰: Y축 기준 -90도 회전 (시계 방향)
    right_init = np.array([
        [0, 0, 1, 0],   # X축이 원래의 Z축으로
        [0, 1, 0, 0],   # Y축은 그대로
        [-1, 0, 0, 0],  # Z축이 원래의 -X축으로
        [0, 0, 0, 1]
    ])
    
    # 왼쪽 뷰: Y축 기준 90도 회전 (반시계 방향)
    left_init = np.array([
        [0, 0, -1, 0],  # X축이 원래의 -Z축으로
        [0, 1, 0, 0],   # Y축은 그대로
        [1, 0, 0, 0],   # Z축이 원래의 X축으로
        [0, 0, 0, 1]
    ])
    
    # 후면 뷰: Y축 기준 180도 회전
    back_init = np.array([
        [-1, 0, 0, 0],  # X축이 원래의 -X축으로
        [0, 1, 0, 0],   # Y축은 그대로
        [0, 0, -1, 0],  # Z축이 원래의 -Z축으로
        [0, 0, 0, 1]
    ])
    
    # 4. 정렬 순서대로 진행
    # 4.1. 정면 포인트 클라우드를 기준으로 설정
    front_pcd = processed_pcds[0]
    aligned_pcds = [front_pcd]
    
    # 두 포인트 클라우드를 정렬하는 함수 (재사용 가능)
    def align_two_point_clouds(source, target, init_transform=None, threshold=30.0):
        # 초기 변환 적용 (지정된 경우)
        source_aligned = copy.deepcopy(source)
        if init_transform is not None:
            source_aligned.transform(init_transform)
            
        # 글로벌 정렬 (RANSAC)
        print("  Computing FPFH features...")
        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source_aligned,
            o3d.geometry.KDTreeSearchParamHybrid(radius=30, max_nn=100)
        )
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            target,
            o3d.geometry.KDTreeSearchParamHybrid(radius=30, max_nn=100)
        )
        
        print("  Performing global registration...")
        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_aligned, target, source_fpfh, target_fpfh, 
            mutual_filter=True,
            max_correspondence_distance=threshold*2,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=4,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(threshold*2)
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 500)
        )
        
        source_aligned.transform(result_ransac.transformation)
        print(f"  Global registration fitness: {result_ransac.fitness:.4f}")
        
        # 정밀 정렬 (ICP)
        print("  Performing ICP refinement...")
        result_icp = o3d.pipelines.registration.registration_icp(
            source_aligned, target, threshold,
            np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6,
                relative_rmse=1e-6,
                max_iteration=100
            )
        )
        
        source_aligned.transform(result_icp.transformation)
        print(f"  ICP refinement fitness: {result_icp.fitness:.4f}")
        
        return source_aligned
    
    # 4.2. 정면을 기준으로 오른쪽 뷰 정렬
    if len(processed_pcds) > 1:
        print("\nAligning right side view to front view...")
        right_pcd = processed_pcds[1]
        aligned_right = align_two_point_clouds(right_pcd, front_pcd, right_init, distance_threshold)
        aligned_pcds.append(aligned_right)
    
    # 4.3. 정면을 기준으로 왼쪽 뷰 정렬
    if len(processed_pcds) > 2:
        print("\nAligning left side view to front view...")
        left_pcd = processed_pcds[2]
        aligned_left = align_two_point_clouds(left_pcd, front_pcd, left_init, distance_threshold)
        aligned_pcds.append(aligned_left)
    
    # 4.4. 정렬된 정면/좌/우를 기준으로 후면 정렬
    if len(processed_pcds) > 3:
        print("\nAligning back view...")
        back_pcd = processed_pcds[3]
        
        # 4.4.1. 먼저 정면과 대략적으로 정렬
        print("First aligning back view to front view...")
        back_aligned = align_two_point_clouds(back_pcd, front_pcd, back_init, distance_threshold)
        
        # 4.4.2. 정면, 좌, 우를 통합한 타겟 생성
        print("Creating combined target point cloud...")
        combined_target = o3d.geometry.PointCloud()
        all_points = []
        all_colors = []
        
        for aligned_pcd in aligned_pcds:
            all_points.extend(np.asarray(aligned_pcd.points))
            all_colors.extend(np.asarray(aligned_pcd.colors))
        
        combined_target.points = o3d.utility.Vector3dVector(np.array(all_points))
        combined_target.colors = o3d.utility.Vector3dVector(np.array(all_colors))
        combined_target.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30)
        )
        
        # 4.4.3. 통합 타겟에 후면 정밀 정렬
        print("Fine-tuning back view alignment with all other views...")
        result_icp = o3d.pipelines.registration.registration_icp(
            back_aligned, combined_target, distance_threshold,
            np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6,
                relative_rmse=1e-6,
                max_iteration=100
            )
        )
        
        back_aligned.transform(result_icp.transformation)
        print(f"Final back alignment fitness: {result_icp.fitness:.4f}")
        aligned_pcds.append(back_aligned)
    
    # 5. 시각화 옵션이 활성화된 경우 정렬된 포인트 클라우드 표시
    if vis:
        print("\nVisualizing aligned point clouds...")
        o3d.visualization.draw_geometries(aligned_pcds)
    
    # 모든 정렬 결과 반환
    print("Alignment process completed successfully!")
    return aligned_pcds

def combine_point_clouds(pcds):
    """
    여러 포인트 클라우드를 결합하여 하나의 포인트 클라우드로 만듭니다.
    중복 제거 및 다운샘플링을 통해 최적화합니다.
    """
    print("Combining aligned point clouds...")
    
    # 1. 모든 포인트 클라우드를 하나로 합치기
    merged_cloud = o3d.geometry.PointCloud()
    points = []
    colors = []
    
    for i, pcd in enumerate(pcds):
        # 각 포인트 클라우드의 포인트와 색상을 추출
        pcd_points = np.asarray(pcd.points)
        pcd_colors = np.asarray(pcd.colors)
        
        # 유효한 포인트만 추가 (NaN이나 무한대 제거)
        valid_indices = np.where(np.all(np.isfinite(pcd_points), axis=1))[0]
        if len(valid_indices) < len(pcd_points):
            print(f"Removed {len(pcd_points) - len(valid_indices)} invalid points from cloud {i}")
        
        # 각 포인트 클라우드에서 동일한 비율의 포인트 추출 (균형있는 결합)
        valid_points = pcd_points[valid_indices]
        valid_colors = pcd_colors[valid_indices]
        
        # 각 뷰에서 비슷한 수의 포인트를 사용하도록 조정
        target_points_per_view = 10000
        if len(valid_points) > target_points_per_view:
            indices = np.random.choice(len(valid_points), target_points_per_view, replace=False)
            valid_points = valid_points[indices]
            valid_colors = valid_colors[indices]
        
        points.extend(valid_points)
        colors.extend(valid_colors)
        print(f"Added {len(valid_points)} points from view {i}")
    
    merged_cloud.points = o3d.utility.Vector3dVector(np.array(points))
    merged_cloud.colors = o3d.utility.Vector3dVector(np.array(colors))
    
    print(f"Raw combined point cloud has {len(merged_cloud.points)} points")
    
    # 2. 복셀 다운샘플링 (중복 제거)
    print("Applying voxel downsampling...")
    voxel_size = 2.0  # 적당한 복셀 크기로 중복 제거
    merged_cloud = merged_cloud.voxel_down_sample(voxel_size=voxel_size)
    print(f"After voxel downsampling: {len(merged_cloud.points)} points")
    
    # 3. 통계적 이상치 제거
    print("Removing outliers...")
    try:
        cl, ind = merged_cloud.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.5)
        merged_cloud = cl
        print(f"After statistical outlier removal: {len(merged_cloud.points)} points")
    except Exception as e:
        print(f"Warning: Statistical outlier removal failed: {e}")
    
    # 4. 반경 기반 이상치 제거 (고립된 포인트 제거)
    try:
        cl, ind = merged_cloud.remove_radius_outlier(nb_points=16, radius=10.0)
        merged_cloud = cl
        print(f"After radius outlier removal: {len(merged_cloud.points)} points")
    except Exception as e:
        print(f"Warning: Radius outlier removal failed: {e}")
    
    # 5. 법선 벡터 계산 및 최적화
    print("Estimating normals for combined point cloud...")
    try:
        # 더 높은 품질의 법선 계산
        merged_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30)
        )
        
        # 법선 방향 최적화
        merged_cloud.orient_normals_consistent_tangent_plane(k=30)
        print("Normal estimation successful")
    except Exception as e:
        print(f"Warning: Normal estimation failed: {e}")
        # 대체 법선 추정 방법
        try:
            print("Trying alternative normal estimation...")
            merged_cloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=20, max_nn=20)
            )
            print("Alternative normal estimation successful")
        except:
            print("All normal estimation attempts failed")
    
    print(f"Final combined point cloud has {len(merged_cloud.points)} points")
    return merged_cloud

def estimate_normals(pcd):
    """
    포인트 클라우드의 법선 벡터를 추정합니다.
    적절한 파라미터를 사용하여 정확한 법선 계산을 시도합니다.
    """
    print("Estimating normals...")
    try:
        # 법선 벡터 추정
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30)
        )
        
        # 법선 벡터 일관성 향상
        pcd.orient_normals_consistent_tangent_plane(k=30)
        
        print("Normal estimation successful")
    except Exception as e:
        print(f"Warning: Normal estimation failed: {e}")
    
    return pcd

def create_mesh(pcd):
    """
    포인트 클라우드에서 메시를 생성합니다.
    Poisson 알고리즘을 이용하여 3D 메시를 생성합니다.
    """
    print("Creating mesh using Poisson reconstruction...")
    
    try:
        # Poisson 알고리즘으로 메시 생성 (깊이 파라미터 조정)
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, 
            depth=9,
            width=0,
            scale=1.1,
            linear_fit=False
        )
        
        # 저밀도 영역 제거 (퀀타일 임계값 조정)
        density_threshold = np.quantile(densities, 0.05)  # 하위 5% 밀도 제거
        vertices_to_remove = densities < density_threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        # 결과 확인
        print(f"Mesh created with {len(mesh.triangles)} triangles and {len(mesh.vertices)} vertices")
        
        # 메시 정제 및 최적화
        mesh.compute_vertex_normals()
        
        # 메시 단순화 (필요한 경우)
        if len(mesh.triangles) > 100000:
            print("Simplifying mesh...")
            mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=100000)
            print(f"Simplified mesh has {len(mesh.triangles)} triangles")
        
        return mesh
    
    except Exception as e:
        print(f"Error in mesh creation: {e}")
        
        # 대체 메시 생성 방법 시도
        print("Trying alternative mesh creation method (Ball Pivoting)...")
        try:
            # 볼 피봇팅 알고리즘으로 메시 생성
            radii = [10, 20, 30, 40]  # 다양한 반경 시도
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector(radii)
            )
            
            # 후처리
            mesh.compute_vertex_normals()
            print(f"Ball pivoting mesh created with {len(mesh.triangles)} triangles")
            return mesh
        
        except Exception as e2:
            print(f"Alternative mesh creation also failed: {e2}")
            
            # 최후의 수단: 빈 메시 반환
            print("Returning empty mesh")
            return o3d.geometry.TriangleMesh()

def process_depth_maps(front_path, back_path, right_path, left_path, output_dir=None):
    """
    4개의 깊이 맵을 처리하여 3D 모델을 생성합니다.
    """
    # 출력 디렉토리 설정
    if output_dir is None:
        output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 깊이 맵 로드
    depth_maps = []
    paths = [front_path, back_path, right_path, left_path]
    labels = ["정면", "후면", "오른쪽", "왼쪽"]
    
    for path, label in zip(paths, labels):
        try:
            depth_array = load_depth_map(path)
            depth_maps.append((depth_array, label))
        except Exception as e:
            print(f"Error loading {label} depth map: {e}")
    
    if len(depth_maps) == 0:
        raise ValueError("No valid depth maps loaded")
    
    # 포인트 클라우드 생성
    point_clouds = []
    for depth_array, label in depth_maps:
        pcd = create_point_cloud_from_depth(depth_array, view=label)
        point_clouds.append(pcd)
    
    # 포인트 클라우드 정렬
    aligned_pcds = align_point_clouds(point_clouds, vis=True)
    
    # 포인트 클라우드 결합
    combined_pcd = combine_point_clouds(aligned_pcds)
    
    # 법선 벡터 추정
    combined_pcd = estimate_normals(combined_pcd)
    
    # 메시 생성
    mesh = create_mesh(combined_pcd)
    
    # 결과 저장
    output_pcd_path = os.path.join(output_dir, "combined_point_cloud.ply")
    output_mesh_path = os.path.join(output_dir, "3d_model.obj")
    
    o3d.io.write_point_cloud(output_pcd_path, combined_pcd)
    o3d.io.write_triangle_mesh(output_mesh_path, mesh)
    
    print(f"Point cloud saved to: {output_pcd_path}")
    print(f"Mesh saved to: {output_mesh_path}")
    
    # 결과 시각화
    o3d.visualization.draw_geometries([combined_pcd])
    o3d.visualization.draw_geometries([mesh])
    
    return combined_pcd, mesh

def main():
    # 파일 경로 설정
    # front_path = "test/정상/정면_여/DepthMap17.bmp"
    # back_path = "test/정상/후면_여/DepthMap17.bmp"
    # right_path = "test/정상/오른쪽_여/DepthMap17.bmp"
    # left_path = "test/정상/왼쪽_여/DepthMap17.bmp"

    front_path = "test/정상/정면_남/DepthMap0.bmp"
    back_path = "test/정상/후면_남/DepthMap0.bmp"
    right_path = "test/정상/오른쪽_남/DepthMap0.bmp"
    left_path = "test/정상/왼쪽_남/DepthMap0.bmp"


    # 출력 디렉토리 설정
    output_dir = "output/3d_models"
    
    # 깊이 맵 처리 및 3D 모델 생성
    try:
        pcd, mesh = process_depth_maps(front_path, back_path, right_path, left_path, output_dir)
        print("3D model creation completed successfully!")
    except Exception as e:
        print(f"Error creating 3D model: {e}")

if __name__ == "__main__":
    main()
