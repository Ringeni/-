# 多摄像头跟踪系统后端

基于 FastAPI 的多摄像头目标跟踪后端服务。

## 技术栈

- **Web 框架**: FastAPI + Uvicorn
- **数据库**: MySQL (异步 aiomysql + SQLAlchemy 2.0)
- **缓存**: Redis
- **认证**: JWT (python-jose)
- **数据库迁移**: Alembic

## 快速启动

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
# Windows
copy .env.example .env

# Linux / macOS
cp .env.example .env
```

编辑 `.env` 文件，配置以下必填项：

| 变量名 | 说明 | 示例 |
|--------|------|------|
| MYSQL_HOST | MySQL 服务器地址 | localhost |
| MYSQL_PORT | MySQL 端口 | 3306 |
| MYSQL_USER | MySQL 用户名 | root |
| MYSQL_PASSWORD | MySQL 密码 | your_password |
| MYSQL_DB | 数据库名 | tracking_db |
| REDIS_HOST | Redis 服务器地址 | localhost |
| REDIS_PORT | Redis 端口 | 6379 |
| REDIS_DB | Redis 数据库编号 | 0 |
| JWT_SECRET_KEY | JWT 签名密钥 | 生成一个随机字符串 |

**生成 JWT_SECRET_KEY 的方法：**

```bash
# Python
python -c "import secrets; print(secrets.token_hex(32))"
```

### 3. 初始化数据库

```bash
alembic upgrade head
```

### 4. 启动服务

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

服务启动后访问 http://localhost:8000/docs 查看 API 文档。

## 创建迁移

```bash
alembic migrate -m "描述"
```

迁移文件位于 `alembic/versions/` 目录。

## 项目结构

```
backend/
├── app/
│   ├── api/          # API 路由 (auth, users, cameras, tracks, warnings, monitor)
│   ├── core/         # 核心模块 (config, database, redis, security, errors)
│   ├── services/     # 业务逻辑
│   ├── schemas/      # Pydantic 模型
│   ├── models/       # SQLAlchemy 模型
│   └── main.py       # 应用入口
├── alembic/          # 数据库迁移
└── .env.example      # 环境变量示例
```

## API 概览

- **基础路径**: `/api/v1`
- **认证**: JWT 令牌
- **功能模块**:
  - 用户管理 (注册/登录)
  - 摄像头管理
  - 目标跟踪
  - 轨迹数据
  - 告警管理
  - 实时监控 (WebSocket)
  - 健康检查

## 核心约定

- 统一异常处理: `AppException` (见 `app/core/errors.py`)
- 配置管理: pydantic-settings (见 `app/core/config.py`)
- 异步数据库操作: SQLAlchemy async + aiomysql