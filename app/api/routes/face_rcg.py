from fastapi import APIRouter


router = APIRouter()


@router.post("/start")
def start_record(): ...


@router.post("/stop")
def stop_record(): ...
