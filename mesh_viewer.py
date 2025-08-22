import open3d as o3d
import numpy as np
import os

def view_mesh(mesh_path):
    """
    생성된 메시 파일을 시각화합니다.
    """
    if not os.path.exists(mesh_path):
        print(f"메시 파일을 찾을 수 없습니다: {mesh_path}")
        return
    
    print(f"메시 로딩 중: {mesh_path}")
    
    # 메시 로드
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    
    if len(mesh.vertices) == 0:
        print("메시 로드 실패!")
        return
    
    print(f"로드된 메시 정보:")
    print(f"- 정점 수: {len(mesh.vertices)}")
    print(f"- 삼각형 수: {len(mesh.triangles)}")
    print(f"- 색상 정보: {'있음' if mesh.has_vertex_colors() else '없음'}")
    
    # 법선 벡터 계산 (렌더링 품질 향상)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    
    # 색상이 없으면 기본 색상 적용
    if not mesh.has_vertex_colors():
        mesh.paint_uniform_color([0.7, 0.7, 0.7])  # 회색
    
    # 시각화
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Generated Mesh Viewer", width=1024, height=768)
    vis.add_geometry(mesh)
    
    # 렌더링 옵션 설정
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.1, 0.1, 0.1])  # 어두운 회색 배경
    opt.mesh_show_wireframe = False
    opt.mesh_show_back_face = True
    
    # 카메라 위치 설정
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    ctr.set_front([0.5, -0.5, -0.5])
    ctr.set_up([0, -1, 0])
    
    print("메시 시각화 시작 (창을 닫으면 종료됩니다)")
    vis.run()
    vis.destroy_window()

def compare_mesh_and_pointcloud():
    """
    생성된 메시를 시각화합니다. (단순화된 버전)
    """
    mesh_path = "output/3d_models/body_mesh.ply"
    
    if not os.path.exists(mesh_path):
        print(f"메시 파일을 찾을 수 없습니다: {mesh_path}")
        return
    
    # 메시 로드
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    
    # 메시에 색상 적용
    if not mesh.has_vertex_colors():
        mesh.paint_uniform_color([0.8, 0.8, 0.9])  # 연한 파란색
    
    # 시각화
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Body Mesh", width=1200, height=800)
    
    # 메시 추가
    vis.add_geometry(mesh)
    
    # 렌더링 옵션
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.mesh_show_wireframe = False
    opt.mesh_show_back_face = True
    
    # 카메라 설정
    ctr = vis.get_view_control()
    ctr.set_zoom(0.7)
    
    print("3D 신체 메시 시각화")
    vis.run()
    vis.destroy_window()

def view_mesh_info():
    """
    생성된 메시의 상세 정보를 출력합니다.
    """
    mesh_paths = [
        "output/3d_models/body_mesh.obj",
        "output/3d_models/body_mesh.ply"
    ]
    
    for mesh_path in mesh_paths:
        if os.path.exists(mesh_path):
            print(f"\n=== {mesh_path} ===")
            mesh = o3d.io.read_triangle_mesh(mesh_path)
            
            print(f"정점 수: {len(mesh.vertices)}")
            print(f"삼각형 수: {len(mesh.triangles)}")
            print(f"색상 정보: {'있음' if mesh.has_vertex_colors() else '없음'}")
            print(f"법선 벡터: {'있음' if mesh.has_vertex_normals() else '없음'}")
            
            # 바운딩 박스 정보
            bbox = mesh.get_axis_aligned_bounding_box()
            print(f"바운딩 박스 크기: {bbox.get_extent()}")
            print(f"중심점: {bbox.get_center()}")
            
            # 파일 크기
            file_size = os.path.getsize(mesh_path) / 1024  # KB
            print(f"파일 크기: {file_size:.2f} KB")
        else:
            print(f"파일을 찾을 수 없습니다: {mesh_path}")

if __name__ == "__main__":
    print("=== 메시 뷰어 ===")
    print("1. 메시 정보 확인")
    print("2. 메시 시각화")
    print("3. 메시 뷰어")
    
    choice = input("선택하세요 (1-3): ")
    
    if choice == "1":
        view_mesh_info()
    elif choice == "2":
        mesh_path = "output/3d_models/body_mesh.ply"
        view_mesh(mesh_path)
    elif choice == "3":
        compare_mesh_and_pointcloud()
    else:
        print("잘못된 선택입니다.")
