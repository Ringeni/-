import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

async def add_user():
    engine = create_async_engine("mysql+aiomysql://root:123456@127.0.0.1:3306/tracking")
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as session:
        from passlib.context import CryptContext
        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        password_hash = pwd_context.hash("test123456")
        
        from sqlalchemy import text
        await session.execute(
            text("INSERT INTO tracking.user (username, password_hash, email, role) VALUES (:username, :password_hash, :email, :role)"),
            {"username": "testuser", "password_hash": password_hash, "email": "test@test.com", "role": "viewer"}
        )
        await session.commit()
        print("User created!")
    
    await engine.dispose()

asyncio.run(add_user())