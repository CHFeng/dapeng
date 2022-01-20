import json
import requests

from typing import Optional
from datetime import datetime as dt
from uuid import UUID
from fastapi import FastAPI, HTTPException
from fastapi.openapi.utils import get_openapi
from db_postgres import Database


APP_VERSION = "0.1.0"
# 建立一個 Fast API application
app = FastAPI()
# create postgres instance
db = Database()


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="大鵬灣車流&人流計數系統",
        version=APP_VERSION,
        description="此文件說明如何透過Restful API存取車流&人流計數的統計資料與RTSP相關設定值的設定與存取",
        routes=app.routes,
    )
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema


# set custom openapi
app.openapi = custom_openapi


@app.get("/")
def check_health():
    '''
    檢查相關服務狀態\n
    version: 此服務的版本號碼\n
    nvrStatus: 從NVR主機回傳的Http回應狀態碼\n
    nvrHosts: NVR主機名稱\n
    '''

    data = {
        'version': APP_VERSION,
        'nvrStatus': None,
        'nvrHosts': []
    }
    try:
        # get nvr config
        config = db.get_config()
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))

    try:
        url = "http://{}:{}/hosts/".format(config['host'], config['port'])
        # get domain hosts
        result = requests.get(url, auth=(
            config['account'], config['password']))
        data["nvrStatus"] = result.status_code
        if result.status_code == requests.codes.ok:
            data["nvrHosts"] = json.loads(result.text)
    except Exception as err:
        data["nvrStatus"] = 500
        data["detail"] = str(err)

    return data


@app.get("/check_nvr")
def check_nvr():
    '''
    檢查NVR影像來源狀態\n
    nvrStatus: 從NVR主機回傳的Http回應狀態碼\n
    resp
    '''

    data = {
        'nvrStatus': None,
        'resp': {}
    }
    try:
        # get nvr config
        config = db.get_config()
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))

    try:
        url = "http://{}:{}/video-origins/".format(
            config['host'], config['port'])
        # get video source status
        result = requests.get(url, auth=(
            config['account'], config['password']))
        data["nvrStatus"] = result.status_code
        if result.status_code == requests.codes.ok:
            data["resp"] = json.loads(result.text)
    except Exception as err:
        data["nvrStatus"] = 500
        data["detail"] = str(err)

    return data


@app.get("/rtsp_config")
def get_rtsp_config():
    '''
    取得NVR主機的設定值,帳號,密碼,連線位址,PORT
    '''

    config = {}
    try:
        config = db.get_config()
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))
    else:
        return config


@app.patch("/rtsp_config")
def update_config(account: Optional[str] = None, password: Optional[str] = None, host: Optional[str] = None, port: Optional[int] = None):
    '''
    更新NVR主機的設定值
    '''

    config = {}
    try:
        config = db.update_config(account, password, host, port)
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))
    else:
        return config


@app.get("/records")
def get_records(id: Optional[UUID] = None, cam_id: Optional[str] = None, start_time: Optional[dt] = None, end_time: Optional[dt] = None, type: Optional[str] = None):
    '''
    取得指定影像來源、日期與時間參數下的物件辨識統計資料\n
    沒有設定參數時取得全部資料\n
    有設定時間時,start_time 與 end_time皆須設定\n
    start_time & end_time format are "%Y-%m-%d %H:%M:%S".\n
    e.g.: 2022-01-19 08:00:00 \n
    Response Value:\n
    statistics: 每個影像來源的統計數據\n
    records: 指定條件下的每筆紀錄\n
    '''

    data = {
        'statistics': {},
        'records': []
    }
    if start_time and not end_time:
        raise HTTPException(
            status_code=422, detail="end_time could not be empty")
    if end_time and not start_time:
        raise HTTPException(
            status_code=422, detail="start_time could not be empty")
    if start_time and end_time and start_time > end_time:
        raise HTTPException(
            status_code=422, detail="start_time should be less than end_time")
    try:
        if not id and not cam_id and not start_time and not end_time and not type:
            rows = db.get_all_records()
        else:
            rows = db.get_record(id, cam_id, start_time, end_time, type)

        data['records'] = rows
        # 統計物件計數數值
        statistics = {}
        for row in rows:
            key = row["camId"]
            type = row["type"]
            # 判斷此型態的物件是否存在
            if key not in statistics:
                statistics[key] = {}
                statistics[key] = {}
            if type not in statistics[key]:
                statistics[key][type] = {'inCounter': 0, 'outCounter': 0}

            # 累加數據
            statistics[key][type]['inCounter'] += row['inCounter']
            statistics[key][type]['outCounter'] += row['outCounter']

        data["statistics"] = statistics

    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))
    else:
        return data


@app.post("/record")
def add_record(camId: str, time: dt, type: str, inValue: int, outValue: int):
    '''
    新增一筆影像辨識紀錄\n
    time format is "%Y-%m-%d %H:%M:%S".\n
    e.g.: 2022-01-19 08:00:00
    '''

    try:
        db.add_record(camId, time, type, inValue, outValue)
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))
    else:
        return {"detail": "OK"}


@app.delete("/record")
def delect_record(id: UUID):
    '''
    刪除一筆影像紀錄\n
    id為該筆紀錄的uuid
    '''

    try:
        db.delete_record(id)
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))
    else:
        return {"detail": "OK"}
