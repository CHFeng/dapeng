from typing import Optional
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

APP_VERSION = "0.1.0"
app = FastAPI()  # 建立一個 Fast API application


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="大鵬灣車流&人流計數系統",
        version=APP_VERSION,
        description="此文件說明如何透過Restful API存取車流&人流計數的統計資料",
        routes=app.routes,
    )
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema


# set custom openapi
app.openapi = custom_openapi


@app.get("/")  # 回傳此服務的版本號碼
def check_health():
    return "The Service works fine, Version:{}".format(APP_VERSION)


@app.get("/check_nvr")  # 檢查與NVR影像主機間的通訊是否正常
def check_nvr():
    return {"Status": "OK"}


@app.get("/check_ai_model")  # 檢查與NVR影像主機間的通訊是否正常
def check_ai_model():
    return {"Status": "OK"}


@app.get("/object_counter")  # 指定影像來源、日期與時間參數下的物件辨識統計資料
def get_object_counter(cam_id: str, date: str, start_time: str, end_time: str):
    print(cam_id, date, start_time, end_time)
    data = {
        "Big Car": 10,
        "Small Car": 50
    }
    return data
