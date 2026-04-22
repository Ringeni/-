import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

async def update_user():
    engine = create_async_engine("mysql+aiomysql://root:123456@127.0.0.1:3306/tracking")
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as session:
        from sqlalchemy import text
        await session.execute(
            text("UPDATE tracking.user SET password_hash = :hash WHERE username = :username"),
            {"hash": "$pbkdf2-sha256$29000$EUJojfE.53wvpbSWUkppTQ$UdqpRNXZAvwm6IpByX0ueUUokE.GDpefo79pUU9iFcU", "username": "testuser"}
        )
        await session.commit()
        print("User updated!")
    
    await engine.dispose()

asyncio.run(update_user())