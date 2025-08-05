#!/usr/bin/env python3
"""
Higgs Audio v2 前端功能测试脚本

此脚本提供简单的HTTP服务器来测试重构后的前端页面
"""

import http.server
import socketserver
import webbrowser
import threading
import time
import os
from pathlib import Path

def start_frontend_server(port=3000):
    """启动前端静态文件服务器"""
    
    # 确保在正确的目录
    project_root = Path(__file__).parent
    frontend_dir = project_root / "frontend"
    
    if not frontend_dir.exists():
        print(f"错误: 前端目录不存在: {frontend_dir}")
        return
    
    # 切换到frontend目录
    os.chdir(frontend_dir)
    
    class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            # 添加CORS头部以支持跨域请求
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            super().end_headers()
    
    try:
        with socketserver.TCPServer(("", port), MyHTTPRequestHandler) as httpd:
            print(f"前端服务器启动成功!")
            print(f"访问地址: http://localhost:{port}")
            print("\n可用页面:")
            print(f"  - 主页 (所有功能): http://localhost:{port}/index.html")
            print(f"  - 基础生成: http://localhost:{port}/generate.html") 
            print(f"  - 语音克隆: http://localhost:{port}/voice-clone.html")
            print("\n注意:")
            print("  1. 确保后端服务器在 http://localhost:8000 运行")
            print("  2. 如需测试完整功能，请先启动后端服务")
            print("  3. 按 Ctrl+C 停止服务器")
            print("\n服务器状态: 运行中...")
            
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n服务器已停止")
    except Exception as e:
        print(f"启动服务器失败: {e}")

def check_backend_status():
    """检查后端服务器状态"""
    import urllib.request
    import json
    
    try:
        with urllib.request.urlopen('http://localhost:8000/health', timeout=5) as response:
            data = json.loads(response.read().decode())
            print("✅ 后端服务器状态:")
            print(f"   状态: {data.get('status', 'unknown')}")
            print(f"   模型已加载: {data.get('model_loaded', False)}")
            print(f"   设备: {data.get('device', 'unknown')}")
            print(f"   可用语音数: {data.get('voices_available', 0)}")
            return True
    except Exception as e:
        print("❌ 后端服务器连接失败:")
        print(f"   错误: {e}")
        print("   请确保后端服务器在 http://localhost:8000 运行")
        return False

def open_browser(port=3000, delay=2):
    """延迟打开浏览器"""
    time.sleep(delay)
    webbrowser.open(f'http://localhost:{port}/index.html')

if __name__ == "__main__":
    print("Higgs Audio v2 前端测试服务器")
    print("=" * 50)
    
    # 检查后端状态
    print("\n1. 检查后端服务器状态...")
    backend_ok = check_backend_status()
    
    if not backend_ok:
        print("\n提示: 如需完整功能测试，请先启动后端服务:")
        print("   python main.py --host 0.0.0.0 --port 8000")
    
    print("\n2. 启动前端服务器...")
    
    # 在后台启动浏览器
    threading.Thread(target=open_browser, daemon=True).start()
    
    # 启动前端服务器
    start_frontend_server(3000)