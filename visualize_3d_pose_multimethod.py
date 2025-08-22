import numpy as np
import open3d as o3d
from PIL import Image
import copy

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
        points = np.stack([x, -y, depth_map * scale * 2], axis=-1)
    elif view == "right":
        points = np.stack([depth_map * scale * 3, -y, -x], axis=-1)
    elif view == "left":
        points = np.stack([-depth_map * scale * 3, -y, x], axis=-1)
    elif view == "back":
        points = np.stack([x, -y, -depth_map * scale * 2], axis=-1)

    threshold = 0.2
    valid_points = points[depth_map > threshold]
    
    if len(valid_points) > 10000:
        indices = np.random.choice(len(valid_points), 10000, replace=False)
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

def create_initial_transformation(source_view, target_view):
    init_transformation = np.eye(4)
    
    if source_view == "left" and target_view == "front":
        angle = np.pi / 2
        init_transformation[0:3, 0:3] = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
    elif source_view == "right" and target_view == "front":
        angle = -np.pi / 2
        init_transformation[0:3, 0:3] = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
    elif source_view == "back" and target_view == "front":
        init_transformation[0:3, 0:3] = np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, -1]
        ])
    
    return init_transformation

def prepare_dataset(source, target, voxel_size=5.0):
    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)

    source_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    target_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    
    return source_down, target_down, source_fpfh, target_fpfh

def align_point_clouds_gicp(source, target, source_view="", target_view="", threshold=50):
    init_transformation = create_initial_transformation(source_view, target_view)
    source_transformed = source.transform(init_transformation)
    
    # GICP 정합 수행
    result = o3d.pipelines.registration.registration_generalized_icp(
        source_transformed, target,
        max_correspondence_distance=threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6,
            relative_rmse=1e-6,
            max_iteration=200
        )
    )
    
    if result.fitness > 0.01:
        return source.transform(np.dot(result.transformation, init_transformation))
    return source

def align_point_clouds_colored_icp(source, target, source_view="", target_view="", threshold=50):
    init_transformation = create_initial_transformation(source_view, target_view)
    source_transformed = source.transform(init_transformation)
    
    # Colored ICP 정합 수행
    result = o3d.pipelines.registration.registration_colored_icp(
        source_transformed, target,
        max_correspondence_distance=threshold,
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6,
            relative_rmse=1e-6,
            max_iteration=100
        )
    )
    
    if result.fitness > 0.01:
        return source.transform(np.dot(result.transformation, init_transformation))
    return source

def align_point_clouds_multi_scale(source, target, source_view="", target_view="", threshold=50):
    init_transformation = create_initial_transformation(source_view, target_view)
    source_transformed = source.transform(init_transformation)
    
    voxel_sizes = [8, 4, 2]  # 다중 스케일
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-6,
        relative_rmse=1e-6,
        max_iteration=100
    )
    
    current_transformation = np.identity(4)
    
    for voxel_size in voxel_sizes:
        radius = voxel_size * 2
        source_down = source_transformed.voxel_down_sample(voxel_size)
        target_down = target.voxel_down_sample(voxel_size)
        
        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
        
        result = o3d.pipelines.registration.registration_icp(
            source_down, target_down,
            max_correspondence_distance=voxel_size * 2,
            init=current_transformation,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=criteria
        )
        current_transformation = result.transformation
        
    final_transformation = np.dot(current_transformation, init_transformation)
    if result.fitness > 0.01:
        return source.transform(final_transformation)
    return source

def align_point_clouds_global(source, target, source_view="", target_view="", threshold=50):
    init_transformation = create_initial_transformation(source_view, target_view)
    source_transformed = source.transform(init_transformation)
    
    voxel_size = 5.0
    source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(
        source_transformed, target, voxel_size)
    
    # Global registration (RANSAC)
    distance_threshold = voxel_size * 1.5
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down,
        source_fpfh, target_fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )
    
    # Local refinement with ICP
    result_icp = o3d.pipelines.registration.registration_icp(
        source_transformed, target,
        threshold,
        result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6,
            relative_rmse=1e-6,
            max_iteration=200
        )
    )
    
    final_transformation = np.dot(result_icp.transformation, init_transformation)
    if result_icp.fitness > 0.01:
        return source.transform(final_transformation)
    return source

def visualize_results(point_clouds, method_name):
    # 모든 포인트 클라우드를 하나로 합치기
    merged_cloud = o3d.geometry.PointCloud()
    points = []
    colors = []
    for pcd in point_clouds:
        points.extend(np.asarray(pcd.points))
        colors.extend(np.asarray(pcd.colors))
    merged_cloud.points = o3d.utility.Vector3dVector(np.array(points))
    merged_cloud.colors = o3d.utility.Vector3dVector(np.array(colors))
    
    # 노이즈 제거 및 다운샘플링
    merged_cloud = merged_cloud.voxel_down_sample(voxel_size=2.0)
    
    # 좌표축 생성
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100)
    
    # 시각화
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"3D Pose Visualization - {method_name}", width=1024, height=768)
    
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

def main():
    views = {
        "front": r"d:\기타\파일 자료\파일\프로젝트 PJ\AAAAAA2_3D_자세\test\정상\정면_남\DepthMap0.bmp",
        "right": r"d:\기타\파일 자료\파일\프로젝트 PJ\AAAAAA2_3D_자세\test\정상\오른쪽_남\DepthMap0.bmp",
        "left": r"d:\기타\파일 자료\파일\프로젝트 PJ\AAAAAA2_3D_자세\test\정상\왼쪽_남\DepthMap0.bmp",
        "back": r"d:\기타\파일 자료\파일\프로젝트 PJ\AAAAAA2_3D_자세\test\정상\후면_남\DepthMap0.bmp"
    }
    
    # 포인트 클라우드 생성
    point_clouds = {}
    for view_name, file_path in views.items():
        depth_map = load_depth_map(file_path)
        if depth_map is not None:
            pcd = create_point_cloud_from_depth(depth_map, view_name)
            if pcd is not None:
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
                point_clouds[view_name] = pcd

    # 각 정합 방법에 대해 실행
    alignment_methods = [
        ("GICP", align_point_clouds_gicp),
        ("Colored ICP", align_point_clouds_colored_icp),
        ("Multi-scale ICP", align_point_clouds_multi_scale),
        ("Global Registration + ICP", align_point_clouds_global)
    ]
    
    for method_name, align_func in alignment_methods:
        print(f"\n적용 중인 방법: {method_name}")
        aligned_clouds = [point_clouds["front"]]
        target = point_clouds["front"]
        
        alignment_order = [
            ("back", 50),
            ("right", 150),
            ("left", 150)
        ]
        
        for view, threshold in alignment_order:
            if view in point_clouds:
                aligned = align_func(
                    copy.deepcopy(point_clouds[view]),
                    target,
                    source_view=view,
                    target_view="front",
                    threshold=threshold
                )
                aligned_clouds.append(aligned)
        
        visualize_results(aligned_clouds, method_name)

if __name__ == "__main__":
    main()
