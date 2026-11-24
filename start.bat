@echo off
chcp 65001 >nul
title Enterprise RAG Knowledge Base
echo ========================================================
echo   企业级多源智能知识库与文档生成系统 (Enterprise RAG)
echo ========================================================
echo.
echo [1/2] 正在检查与配置运行环境（自动读取项目根目录 .env）...
echo [2/2] 正在启动 Web 服务并打开浏览器...
echo 访问地址: http://localhost:8501
echo.
start http://localhost:8501
python -m streamlit run app.py --server.port 8501
pause
