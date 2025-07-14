#!/usr/bin/env python3
"""
로컬 개발용 GPT 서버 실행 스크립트
"""

import uvicorn
from main import app

if __name__ == "__main__":
    print("🚀 GPT 서버를 시작합니다...")
    print("📍 서버 주소: http://localhost:8001")
    print("📚 API 문서: http://localhost:8001/docs")
    
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8001,
        reload=True,  # 개발용 자동 리로드
        log_level="info"
    ) 