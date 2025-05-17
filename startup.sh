#!/bin/bash

# 로그 파일 경로 설정
LOG_FILE="/workspace/output/system.log"

# output 디렉토리 생성 및 로그 파일 초기화
mkdir -p /workspace/output
rm -f "$LOG_FILE"
touch "$LOG_FILE"

# 로그 함수 정의
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

# 시작 로그 기록
log_message "=== 스크립트 실행 시작 ==="

# 프로젝트 디렉토리로 이동
cd /workspace
log_message "작업 디렉토리로 이동 완료"

# AI_Lambda 디렉토리 생성 및 이동
mkdir -p AI_Lambda
cd AI_Lambda
log_message "AI_Lambda 디렉토리 생성 및 이동 완료"

# 기존 파일 삭제 (숨김 파일 포함)
rm -rf .[!.]* *
log_message "AI_Lambda 디렉토리 내 모든 파일(숨김 파일 포함) 삭제 완료"

# AI_Lambda 레포지토리 클론
git clone -b master https://github.com/OptiQuantTeam/AI_Lambda.git .
log_message "AI_Lambda 레포지토리 클론 완료"

# 상위 디렉토리로 이동하여 AI 학습 실행
cd ..
log_message "AI 학습 시작"
python3 src/main.py
log_message "AI 학습 완료"

# model 디렉토리 생성 및 기존 파일 삭제
mkdir -p AI_Lambda/model
rm -rf AI_Lambda/model/*
log_message "AI_Lambda/model 디렉토리 내 기존 파일 삭제 완료"

# 학습된 모델 파일을 AI_Lambda/model 디렉토리로 복사
cp -r saved_model/* AI_Lambda/model/
cp -r saved_model/metadata/* output/metadata/
log_message "모델 파일 복사 완료"


# AI_Lambda 디렉토리로 이동
cd AI_Lambda

# 변경사항 커밋 및 푸시
if git add . && git commit -m "Update model files" && git push origin master; then
    log_message "모델 파일이 성공적으로 업로드되었습니다."
    # 서버 종료
    log_message "시스템 종료 시작"
    #sudo shutdown -h now
else
    log_message "AI_Lambda 레포지토리 Git 작업 중 오류가 발생했습니다."
    exit 1
fi

# 변경사항 커밋 및 푸시 후 AI_Lambda 디렉토리 내 모든 파일(숨김 파일 포함) 삭제
rm -rf .[!.]*
cd ..
rm -rf AI_Lambda/
log_message "AI_Lambda 디렉토리 내 모든 파일(숨김 파일 포함) 삭제 완료"

# AI_Train 레포지토리 변경사항 커밋 및 푸시
if git add . && git commit -m "Update training results" && git push origin master; then
    log_message "AI_Train 레포지토리 변경사항이 성공적으로 업로드되었습니다."
else
    log_message "AI_Train 레포지토리 Git 작업 중 오류가 발생했습니다."
    exit 1
fi
