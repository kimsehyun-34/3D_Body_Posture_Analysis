import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter

def create_point_cloud_from_depth(depth_map, view):
    if depth_map is None:
        return None
    
    # 가우시안 블러로 노이즈 제거
    depth_map = gaussian_filter(depth_map, sigma=1.0)
    
    size = depth_map.shape[0]
    y, x = np.mgrid[0:size, 0:size]

class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        
        return x

class DCP(nn.Module):
    def __init__(self):
        super(DCP, self).__init__()
        self.encoder = PointNet()
        
    def forward(self, src, tgt):
        src_embedding = self.encoder(src)
        tgt_embedding = self.encoder(tgt)
        return src_embedding, tgt_embedding

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
    
    # 가우시안 블러로 노이즈 제거
    from scipy.ndimage import gaussian_filter
    depth_map = gaussian_filter(depth_map, sigma=1.0)
    
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

    threshold = 0.2
    valid_points = points[depth_map > threshold]
    
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
    
    return pcd

def create_initial_transformation(source_view, target_view="front"):
    # 초기 변환 행렬 설정
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

def align_point_clouds_dcp_icp(source, target, dcp_model, source_view="", target_view="front", threshold=50):
    # 초기 변환 적용
    init_transformation = create_initial_transformation(source_view, target_view)
    source_initial = source.transform(init_transformation)
    
    # FPFH 특징 계산
    source_initial.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))
    target.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))
    
    # DCP를 사용한 정렬
    device = next(dcp_model.parameters()).device
    source_points = torch.FloatTensor(np.asarray(source_initial.points)).transpose(0, 1).unsqueeze(0).to(device)
    target_points = torch.FloatTensor(np.asarray(target.points)).transpose(0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        src_embedding, tgt_embedding = dcp_model(source_points, target_points)
        src_embedding = src_embedding.cpu()
        tgt_embedding = tgt_embedding.cpu()
    
    # 특징 매칭을 통한 대응점 찾기
    dist = torch.cdist(src_embedding, tgt_embedding)
    values, matches = torch.min(dist, dim=1)
    values = values.numpy()
    matches = matches.numpy()
    
    # 거리 기반 필터링 - 상위 50%의 매칭만 사용
    threshold = np.percentile(values, 75)  # 상위 75% 까지 허용
    valid_matches = values < threshold
    
    # 대응점 쌍 생성
    source_indices = np.arange(len(matches))[valid_matches]
    target_indices = matches[valid_matches]
    
    # 충분한 대응점이 있는지 확인
    if len(source_indices) < 3:
        print(f"Not enough correspondences found (found {len(source_indices)})")
        return source_initial  # 초기 변환만 적용된 상태 반환
    
    # Open3D의 대응점 형식으로 변환
    correspondences = np.vstack((source_indices, target_indices)).T
    
    # RANSAC으로 이상치 제거 및 변환 행렬 추정
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        source, target,
        o3d.utility.Vector2iVector(correspondences),
        max_correspondence_distance=threshold * 0.05,  # 임계값 조정
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        ransac_n=4,  # RANSAC 샘플 수 증가
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
    )
    
    # ICP로 미세 조정
    result_icp = o3d.pipelines.registration.registration_icp(
        source, target,
        threshold,
        result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6,
            relative_rmse=1e-6,
            max_iteration=100
        )
    )
    
    if result_icp.fitness > 0.01:
        return source.transform(result_icp.transformation)
    return source

def visualize_3d_pose():
    # DCP 모델 초기화
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dcp_model = DCP().to(device)
    dcp_model.eval()
    
    print("DCP 모델 초기화 완료")
    
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
    
    aligned_clouds = [point_clouds["front"]]
    target = point_clouds["front"]
    
    alignment_order = [
        ("back", 50),
        ("right", 150),
        ("left", 150)
    ]
    
    for view, threshold in alignment_order:
        if view in point_clouds:
            aligned = align_point_clouds_dcp_icp(
                point_clouds[view],
                target,
                dcp_model,
                source_view=view,
                target_view="front",
                threshold=threshold
            )
            aligned_clouds.append(aligned)
    
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
    
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Pose Visualization (DCP + ICP)", width=1024, height=768)
    
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
