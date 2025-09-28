#!/bin/bash

# 무한 루프를 통해 파이썬 스크립트를 계속 실행합니다.
while true
do
    # agentic_main.py를 실행합니다.
    python agentic_main.py
    
    # 파이썬 스크립트의 종료 코드를 확인합니다.
    exit_code=$?
    
    # 종료 코드가 10이면 (우리가 설정할 '예약된 재시작' 코드),
    # 5초간 대기 후 루프를 계속하여 스크립트를 다시 시작합니다.
    if [ $exit_code -eq 10 ]; then
        echo "예약된 재시작을 수행합니다. 5초 후에 다음 학습 세션을 시작합니다..."
        sleep 5
    else
        # 다른 종료 코드(예: 실제 오류)가 발생하면 루프를 중단합니다.
        echo "오류 또는 수동 종료로 인해 학습 루프를 중단합니다. 종료 코드: $exit_code"
        break
    fi
done