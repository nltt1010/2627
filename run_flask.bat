@echo off
chcp 65001 > nul
title VietHerb-AI Server (Flask)

echo ========================================================
echo        KHỞI ĐỘNG HỆ THỐNG VIETHERB-AI (FLASK)
echo ========================================================
echo Đang kích hoạt môi trường Conda...

call conda activate 2627
if %errorlevel% neq 0 (
    echo [LỖI] Không thể kích hoạt môi trường conda '2627'.
    pause
    exit /b
)

echo Môi trường đã sẵn sàng! Đang khởi chạy máy chủ...
echo Truy cập ứng dụng tại: http://localhost:8000
echo.

python -m src.web.app
pause