#!/bin/bash
# Simple EC2 deployment script for Trendy Lyrics API

set -e

echo "🚀 Trendy Lyrics API 배포 시작"
echo "================================"

# 서버 설정
EC2_HOST="3.36.70.96"
KEY_FILE="/Users/kang-youngmin/Desktop/keypair/umc-hackathon.pem"
REMOTE_DIR="/opt/trendy-lyrics"

# 코드 패키징
echo "📦 코드 패키징..."
tar -czf trendy-lyrics-deploy.tar.gz \
    --exclude="*/__pycache__" \
    --exclude="*.pyc" \
    --exclude=".git" \
    --exclude="data/" \
    --exclude=".venv/" \
    --exclude="scripts/" \
    app/ \
    pyproject.toml \
    README.md

echo "✅ 패키징 완료: $(du -h trendy-lyrics-deploy.tar.gz | cut -f1)"

# 서버 배포
echo "📤 서버 배포..."
scp -i $KEY_FILE trendy-lyrics-deploy.tar.gz ec2-user@$EC2_HOST:/tmp/

ssh -i $KEY_FILE ec2-user@$EC2_HOST "
    set -e
    cd $REMOTE_DIR
    
    # 백업 생성
    if [ -d current ]; then
        BACKUP_DIR=\"backup-\$(date +%Y%m%d%H%M%S)\"
        echo \"📋 기존 버전 백업: \$BACKUP_DIR\"
        sudo cp -r current \$BACKUP_DIR
    fi
    
    # 새 버전 배포
    echo \"🔄 새 버전 배포...\"
    sudo mkdir -p current
    cd current
    sudo tar -xzf /tmp/trendy-lyrics-deploy.tar.gz
    
    # 가상환경 및 의존성 설치
    if [ ! -d .venv ]; then
        sudo python3 -m venv .venv
    fi
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -e .
    
    # 서비스 재시작
    echo \"🔄 서비스 재시작...\"
    pkill -f uvicorn || true
    sleep 3
    nohup python -m uvicorn app.main:app --host 127.0.0.1 --port 8010 --log-level info > app.log 2>&1 &
    
    # 상태 확인
    sleep 5
    if pgrep -f uvicorn > /dev/null; then
        echo \"✅ 서비스 시작 성공 (PID: \$(pgrep -f uvicorn))\"
        curl -s http://localhost:8010/health || echo \"⚠️ Health check 실패\"
    else
        echo \"❌ 서비스 시작 실패\"
        exit 1
    fi
"

echo "🎉 배포 완료!"
echo "📊 서비스 상태 확인: ./view_logs.sh"

# 로컬 임시파일 정리
rm -f trendy-lyrics-deploy.tar.gz