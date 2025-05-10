import anyio
from functools import wraps
from typing import  Dict, Optional, Tuple
from fastapi import HTTPException, Request, Header, status 
import logging
from dotenv import load_dotenv 
import os

load_dotenv()


class InMemoryKeyStore:
    """
    Manages API keys and their usage quotas in memory.

    get(key)  -> (quota, used) | None
    incr(key) -> (quota, new_used) | None
    """
    def __init__(self) -> None:
        # Load from environment variable API_KEYS=key1:quota1,key2:quota2,...
        env = os.getenv("API_KEYS", "")
        pairs = [p.strip() for p in env.split(",") if p.strip()]
        self._data: Dict[str, Dict[str, int]] = {}
        self._locks: Dict[str, anyio.Lock] = {}
        for p in pairs:
            try:
                k, q_str = p.split(":")
                q = int(q_str)
                if k and q > 0:
                    self._data[k] = {"quota": q, "used": 0}
                    self._locks[k] = anyio.Lock()
                else:
                    logging.warning(f"Skipping invalid API key entry: '{p}'")
            except ValueError:
                logging.warning(f"Skipping invalid API key format: '{p}'")

        logging.info(f"Loaded {len(self._data)} API keys.")


    async def get(self, k: str) -> Optional[Tuple[int, int]]:
        rec = self._data.get(k)
        # Return a tuple copy to prevent external mutation of 'used'
        return (rec["quota"], rec["used"]) if rec else None

    async def incr(self, k: str) -> Optional[Tuple[int, int]]:
        rec = self._data.get(k)
        lock = self._locks.get(k)
        if rec is None or lock is None:
            return None # Should not happen if key validated first
        async with lock:
            # Check quota again inside lock to handle race conditions
            if rec["used"] < rec["quota"]:
                rec["used"] += 1
            return rec["quota"], rec["used"]

    async def validate(self, key: str, increment: bool = False) -> bool:
        """Checks if key is valid and optionally increments usage if quota allows."""
        quota_info = await self.get(key)
        if not quota_info:
            return False # Invalid key

        quota, used = quota_info
        if used >= quota:
            return False # Quota exceeded

        if increment:
            await self.incr(key) # Increment usage

        return True # Key valid and quota okay

# Dependency for HTTP routes
async def api_key_auth(
    request: Request,
    x_api_key: str = Header(...)
) -> str:
    # Ensure state exists and key_store is accessible
    if not hasattr(request.app.state, 'key_store'):
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Key store not initialized")
    key_store: InMemoryKeyStore = request.app.state.key_store
    quota_info = await key_store.get(x_api_key)

    if not quota_info:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")

    quota, used = quota_info
    if used >= quota:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Quota exceeded")

    # Increment usage count for this request
    await key_store.incr(x_api_key)

    # Return the key, could be used for logging or further checks if needed
    return x_api_key

