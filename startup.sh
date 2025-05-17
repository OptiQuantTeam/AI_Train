#!/bin/bash

# 로그 함수 정의
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

# 로그 파일 경로 설정
LOG_FILE="/workspace/output/system.log"



# output 디렉토리 생성 및 로그 파일 초기화
mkdir -p /workspace/output
mkdir -p /workspace/output/metadata
rm -f "$LOG_FILE"
touch "$LOG_FILE"


# 시작 로그 기록
log_message "=== 스크립트 실행 시작 ==="

# 프로젝트 디렉토리로 이동
cd /workspace
log_message "작업 디렉토리로 이동 완료"


# AI 학습 실행 전에 필요한 디렉토리 생성
mkdir -p /workspace/saved_model/logs
log_message "로그 디렉토리 생성 완료"

# 상위 디렉토리로 이동하여 AI 학습 실행
cd ..
log_message "AI 학습 시작"
# 학습 결과를 로그 파일에 기록
python3 src/main.py 2>&1 | tee -a "$LOG_FILE"
log_message "AI 학습 완료"



# AI_Train 레포지토리 변경사항 커밋 및 푸시
if git add . && git commit -m "Update training results" && git push origin master; then
    log_message "AI_Train 레포지토리 변경사항이 성공적으로 업로드되었습니다."
else
    log_message "AI_Train 레포지토리 Git 작업 중 오류가 발생했습니다."
    exit 1
fi
