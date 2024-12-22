from fastapi import HTTPException, APIRouter


router = APIRouter()


@router.post("/start")
def start_record(): ...


@router.post("/stop")
def stop_record(): ...
