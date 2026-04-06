from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.errors import AppException, ErrorCode
from app.core.security import create_access_token, create_refresh_token
from app.crud.user import UserCRUD
from app.models.user import User
from app.schemas.auth import TokenData

_pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
_settings = get_settings()


class AuthService:
    @staticmethod
    def hash_password(password: str) -> str:
        return _pwd_context.hash(password)

    @staticmethod
    def verify_password(plain_password: str, password_hash: str) -> bool:
        return _pwd_context.verify(plain_password, password_hash)

    @staticmethod
    async def authenticate_user(db: AsyncSession, username: str, password: str) -> User:
        user = await UserCRUD.get_by_username(db, username)
        if user is None or not AuthService.verify_password(password, user.password_hash):
            raise AppException(code=ErrorCode.FORBIDDEN, message="invalid credentials", http_status=401)
        await UserCRUD.update_last_login(db, user)
        return user

    @staticmethod
    def build_tokens(user_id: int) -> TokenData:
        subject = str(user_id)
        access_token = create_access_token(subject=subject)
        refresh_token = create_refresh_token(subject=subject)
        return TokenData(
            access_token=access_token,
            refresh_token=refresh_token,
            expire_hours=_settings.jwt_access_expire_hours,
        )
