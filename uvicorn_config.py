# 🚀 로컬 개발용 uvicorn 설정
import multiprocessing

# CPU 코어 수에 따른 워커 설정
workers = multiprocessing.cpu_count() * 2 + 1

# 성능 최적화 설정
bind = "127.0.0.1:8001"  # 로컬 개발용으로 변경
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
timeout = 120
keepalive = 5

# 로깅 설정
accesslog = "-"
errorlog = "-"
loglevel = "info"

# 프로세스 설정
preload_app = True
daemon = False 