import json
import requests

from typing import List, Optional
from datetime import datetime as dt, timedelta
from uuid import UUID
from fastapi import FastAPI, HTTPException
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel
from db_postgres import Database

APP_VERSION = "1.1.0"
# 建立一個 Fast API application
app = FastAPI()
# create postgres instance
db = Database()


class Record(BaseModel):
    camId: str
    time: dt
    type: str
    inValue: int
    outValue: int


class Records(BaseModel):
    records: List[Record]


# 自定義open API文件說明
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="大鵬灣車流&人流計數系統",
        version=APP_VERSION,
        description="此文件說明如何透過Restful API存取車流&人流計數的統計資料與NVR主機相關設定值的設定與存取",
        routes=app.routes,
    )
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema


# set custom openapi
app.openapi = custom_openapi


# API文件中定義的回傳格式
def resp(errMsg, data=None):
    resp = {"code": "0", "message": ""}

    if errMsg is not None:
        resp["code"] = "1"
        resp["message"] = errMsg
    else:
        resp["data"] = data

    return resp


@app.get("/")
def check_health():
    '''
    檢查相關服務狀態\n
    version: 此服務的版本號碼\n
    nvrStatus: 從NVR主機回傳的Http回應狀態碼\n
    nvrHosts: NVR主機名稱\n
    '''

    data = {'version': APP_VERSION, 'nvrStatus': None, 'nvrHosts': []}
    try:
        # get nvr config
        config = db.get_config()
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))

    try:
        url = "http://{}:{}/hosts/".format(config['host'], config['port'])
        # get domain hosts
        result = requests.get(url,
                              auth=(config['account'], config['password']))
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

    data = {'nvrStatus': None, 'resp': []}
    try:
        # get nvr config
        config = db.get_config()
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))

    try:
        url = "http://{}:{}/video-origins/".format(config['host'],
                                                   config['port'])
        # get video source status
        result = requests.get(url,
                              auth=(config['account'], config['password']))
        data["nvrStatus"] = result.status_code
        if result.status_code == requests.codes.ok:
            resp = json.loads(result.text)
            data["resp"] = list(resp.values())
    except Exception as err:
        data["nvrStatus"] = 500
        data["detail"] = str(err)

    return data


@app.get("/nvr_config")
def get_nvr_config():
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


@app.patch("/nvr_config")
def update_nvr_config(account: Optional[str] = None,
                      password: Optional[str] = None,
                      host: Optional[str] = None,
                      port: Optional[int] = None):
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
def get_records(id: Optional[UUID] = None,
                cam_id: Optional[str] = None,
                start_time: Optional[dt] = None,
                end_time: Optional[dt] = None,
                type: Optional[str] = None):
    '''
    取得指定影像來源、日期與時間參數下的物件辨識統計資料\n
    所有參數皆為optional，沒有設定參數時取得全部資料\n
    有設定時間時,start_time 與 end_time皆須設定\n
    cam_id: NVR影像來源的ID\n
    start_time & end_time format的格式為: "%Y-%m-%d %H:%M:%S".\n
    e.g.: 2022-01-19 08:00:00 \n
    Response Value:\n
    statistics: 每個影像來源的統計數據\n
    records: 指定條件下的每筆紀錄\n
    '''

    data = {'statistics': {}, 'records': []}
    if start_time and not end_time:
        raise HTTPException(status_code=422,
                            detail="end_time could not be empty")
    if end_time and not start_time:
        raise HTTPException(status_code=422,
                            detail="start_time could not be empty")
    if start_time and end_time and start_time > end_time:
        raise HTTPException(status_code=422,
                            detail="start_time should be less than end_time")
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


@app.post("/records")
def add_records(body: Records):
    '''
    新增多筆影像辨識紀錄\n
    camId: 攝影機ID\n
    time: 格式為"%Y-%m-%d %H:%M:%S". e.g.: 2022-01-19 08:00:00\n
    type: 物件型態(person, car, bus...etc)\n
    inValue: 進場次數\n
    outValue: 離場次數\n
    '''
    try:
        for record in body.records:
            db.add_record(record.camId, record.time, record.type,
                          record.inValue, record.outValue)
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))
    else:
        return {"detail": "OK"}


@app.delete("/record")
def delect_record(id: UUID):
    '''
    刪除一筆影像紀錄\n
    id: 該筆紀錄的uuid
    '''

    try:
        db.delete_record(id)
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))
    else:
        return {"detail": "OK"}


@app.post("/camera/records")
def get_records(startTime: int, endTime: int, type: Optional[str] = 'ALL'):
    '''
    查詢某時間段的所有攝影機記錄的資訊\n
    ● startTime(required): 開始時間 (long time to millisecond)\n
    ● endTime(required): 結束時間 (long time to millisecond)\n
    type :\n
        ALL: 全部(列出所有分類)\n
        TRUCK: 大貨車\n
        PICKUP_TRUCK: 小貨車\n
        BUS: 公車\n
        AUTOCAR: 自用車\n
        MOTORCYCLE: 機車\n
        BIKE: 腳踏車\n
        AMBULANCE: 救護車\n
        FIRE_ENGINE: 消防車\n
        POLICE_CAR: 警察車\n
        PEOPLE: 行人\n
    Response說明:\n
    code:執行API結果；0 = 成功，1 = 失敗\n
    message:執行結果；成功=空字串，失敗=錯誤訊息\n
    data: 回傳所有攝影機統計內容；NVR影像來源ID包含該時段的各類型總數\n
    '''

    data = []
    if startTime and endTime and startTime > endTime:
        return resp("startTime should be less than endTime")
    try:
        # conver milliseconds to date type
        start_time = dt.fromtimestamp(startTime / 1000.0)
        end_time = dt.fromtimestamp(endTime / 1000.0)
        # if type is ALL, remove type query
        if type == 'ALL': type = None
        rows = db.get_record(start_time=start_time,
                             end_time=end_time,
                             type=type)

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

        # 轉換資料格式符合需求文件
        for key in statistics.keys():
            row = {"camera": key, "detail": []}
            for objType in statistics[key].keys():
                row["detail"].append({
                    "type":
                    objType,
                    "inCounter":
                    statistics[key][objType]["inCounter"],
                    "outCounter":
                    statistics[key][objType]["outCounter"]
                })
            data.append(row)
    except Exception as err:
        return resp(str(err))
    else:
        return resp(None, data)


@app.get("/camera/traffic")
def get_traffic(camera: str,
                endTime: Optional[int] = int(dt.now().timestamp())):
    '''
    查詢即時路況\n
    ● camera: NVR影像來源ID\n

    備註:\n
    traffic:路況類型\n
        JAMMED:壅塞(車速 < 30 & 車輛計數 > 10)\n
        HEAVY:車多(60 < 車速 > 30 & 10 < 車輛計數 > 3)\n
        LIGHT:順暢(車速 > 60 & 車輛計數 < 3)\n
    '''
    # 取得目前時間前五分鐘內的資料,因AI模組每5分鐘更新資料一次
    end_time = dt.fromtimestamp(endTime)
    start_time = end_time - timedelta(minutes=5)
    data = {"camera": camera, "traffic": "LIGHT"}
    try:
        # TODO 是否指定type=car?
        rows = db.get_record(camId=camera,
                             start_time=start_time,
                             end_time=end_time)
        carCounter = 0
        for row in rows:
            if row["type"] != "person":
                carCounter += row["inCounter"]
        # TODO 根據車速與車輛計數判斷道路狀況,目前缺少車速資訊
        if carCounter > 10:
            data["traffic"] = "JAMMED"
        elif carCounter > 3:
            data["traffic"] = "HEAVY"
        else:
            data["traffic"] = "LIGHT"
    except Exception as err:
        return resp(str(err))
    else:
        return resp(None, data)


@app.get("/camera/statistics/traffic")
def get_statistics_traffic(camera: str, startTime: int, endTime: int):
    '''
    查詢小時路況統計清單\n
    ● camera: NVR影像來源ID\n

    備註:\n
    traffic:路況類型\n
        JAMMED:壅塞(車速 < 30 & 車輛計數 > 10)\n
        HEAVY:車多(60 < 車速 > 30 & 10 < 車輛計數 > 3)\n
        LIGHT:順暢(車速 > 60 & 車輛計數 < 3)\n
    '''
    # conver milliseconds to date type
    start_time = dt.fromtimestamp(startTime / 1000.0)
    end_time = dt.fromtimestamp(endTime / 1000.0)
    data = {"camera": camera, "statisticsList": []}
    try:
        rows = db.get_record(camId=camera,
                             start_time=start_time,
                             end_time=end_time)
        # 從取得的資料中判斷每個小時的道路狀況
        start = start_time
        while True:
            # 設定區間結束時間為區間起始時間+1小時
            end = start + timedelta(hours=1)
            # 如果區間結束時間時間已經超過使用者指定的結束時間,修改區間結束時間設定為使用者指定的結束時間
            if end > end_time:
                end = end_time
            trafficData = {
                "statsTime": start.timestamp() * 1000,
                "camera": camera,
                "traffic": "LIGHT"
            }
            carCounter = 0
            for row in rows:
                if row["type"] != "person" and row["time"] > start and row[
                        "time"] < end:
                    print("Start:{} End:{} data:{}".format(start, end, row))
                    carCounter += row["inCounter"]
            # TODO 根據車速與車輛計數判斷道路狀況,目前缺少車速資訊
            if carCounter > 10:
                trafficData["traffic"] = "JAMMED"
            elif carCounter > 3:
                trafficData["traffic"] = "HEAVY"
            else:
                trafficData["traffic"] = "LIGHT"

            data["statisticsList"].append(trafficData)
            # 執行下一個小時的道路狀況判定
            start = end
            # 已到達使用者指定的結束時間,結束迴圈
            if end == end_time:
                break
    except Exception as err:
        return resp(str(err))
    else:
        return resp(None, data)
