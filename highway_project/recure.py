import zipfile
import os

# 1. 손상된 ZIP 압축 해제
print("📦 TD3 모델 압축 해제 중...")
with zipfile.ZipFile('/Users/tony/Desktop/강화학습/프로젝트/highway_project/models/td3_fixed_seed_1000.zip', 'r') as zip_ref:
    zip_ref.extractall('td3_temp')

# 2. 올바른 구조로 재압축
print("🔄 올바른 구조로 재압축 중...")

# 실제 파일들이 있는 디렉토리 찾기
model_dir = '/Users/tony/Desktop/강화학습/프로젝트/highway_project/models/td3_temp/td3_fixed_seed_1000'

# 새 ZIP 파일 생성
with zipfile.ZipFile('td3_fixed_seed_1000_fixed.zip', 'w', zipfile.ZIP_DEFLATED) as new_zip:
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            file_path = os.path.join(root, file)
            # 루트에 직접 저장 (폴더 구조 제거)
            arcname = file
            new_zip.write(file_path, arcname)
            print(f"  추가: {arcname}")

print("\n✅ 완료!")

# 3. 검증
print("\n🔍 검증 중...")
with zipfile.ZipFile('/Users/tony/Desktop/강화학습/프로젝트/highway_project/models/td3_fixed_seed_1000.zip', 'r') as zf:
    print(f"파일 목록: {zf.namelist()}")


