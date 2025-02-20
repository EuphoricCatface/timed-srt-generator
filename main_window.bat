@echo off

:: venv\Scripts\activate.bat 파일이 있는지 확인
if not exist venv\Scripts\activate.bat (
    echo [오류] venv\Scripts\activate.bat 파일을 찾을 수 없습니다.
    echo README_windows.txt를 참고하여 가상 환경 생성 과정을 먼저 진행해 주세요.
    pause
    exit /b
)

:: 가상 환경 활성화
call venv\Scripts\activate.bat

python main_window.py

deactivate
exit