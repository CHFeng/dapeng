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


@app.get("/")  # 指定 api 路徑 (get方法)
def check_health():
    return "The Service works fine, Version:{}".format(APP_VERSION)


@app.get("/check_nvr")
def check_nvr():
    return {"Status": "OK"}


@app.get("/check_ai_model")
def check_ai_model():
    return {"Status": "OK"}


@app.get("/object_counter")
def get_object_counter(cam_id: str, date: str, time: str):
    print(cam_id, date, time)
    data = {
        "Big Car": 10,
        "Small Car": 50
    }
    return data


@app.get("/users/{user_id}")  # 指定 api 路徑 (get方法)
def read_user(user_id: int, q: Optional[str] = None):
    return {"user_id": user_id, "q": q}
