import json
import requests
import uvicorn

from typing import List, Optional
from datetime import datetime as dt, timedelta
from uuid import UUID
from fastapi import FastAPI, HTTPException
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel
from db_postgres import Database, Record

APP_VERSION = "3.0.0"
POST_ERR_URL = "http://{server_domain}/api/nvr/error"
# 建立一個 Fast API application
app = FastAPI()
# create postgres instance
db = Database()


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
    openapi_schema['info']['x-logo'] = {'url': "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"}
    app.openapi_schema = openapi_schema
    return app.openapi_schema


# set custom openapi
app.openapi = custom_openapi


# 告知斯納捷主機,NVR主機連線錯誤
def send_err_to_web(errorCode: str, errorMsg: str):
    '''
    錯誤記錄通知
    errorCode: 錯誤代碼,可由資策會提供
        A01: NVR主機認證失敗
        A02: NVR主機連線失敗
    errorMsg: 錯誤訊息,真實的錯誤原因
    time: 時間,錯誤發生時間
    source: 來源,iii (固定)

    使用情境如 : 攝影機連線nvr不成功時發送
    '''
    try:
        errCodeList = {'401': 'A01', '404': 'A02', '500': 'A02'}
        if errorCode in errCodeList:
            errorCode = errCodeList[errorCode]
        else:
            errorCode = 'A02'
        body = {'errorCode': errorCode, 'errorMsg': errorMsg, 'time': dt.now().strftime("%Y-%m-%d %H:%M:%S"), 'source': 'iii'}
        print(body)
        return
        result = requests.post(POST_ERR_URL, data=body)
        if result.status_code != requests.codes.ok:
            print("send error to web Err:" + json.loads(result.text))
        else:
            print("post error message to WEB successfully!")
    except Exception as err:
        print("send_err_to_web Err:" + str(err))


# API文件中定義的回傳格式
def resp(errMsg, data=None):
    resp = {'code': "0", 'message': ""}

    if errMsg is not None:
        resp['code'] = "1"
        resp['message'] = errMsg
    else:
        resp['data'] = data

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
        result = requests.get(url, auth=(config['account'], config['password']))
        data['nvrStatus'] = result.status_code
        if result.status_code == requests.codes.ok:
            data['nvrHosts'] = json.loads(result.text)
        else:
            data['detail'] = result.reason
            send_err_to_web(str(result.status_code), result.reason)
    except Exception as err:
        data['nvrStatus'] = 500
        data['detail'] = str(err)
        send_err_to_web('500', str(err))

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
        url = "http://{}:{}/video-origins/".format(config['host'], config['port'])
        # get video source status
        result = requests.get(url, auth=(config['account'], config['password']))
        data['nvrStatus'] = result.status_code
        if result.status_code == requests.codes.ok:
            resp = json.loads(result.text)
            data['resp'] = list(resp.values())
    except Exception as err:
        data['nvrStatus'] = 500
        data['detail'] = str(err)

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
        raise HTTPException(status_code=422, detail="end_time could not be empty")
    if end_time and not start_time:
        raise HTTPException(status_code=422, detail="start_time could not be empty")
    if start_time and end_time and start_time > end_time:
        raise HTTPException(status_code=422, detail="start_time should be less than end_time")
    try:
        if not id and not cam_id and not start_time and not end_time and not type:
            rows = db.get_all_records()
        else:
            rows = db.get_record(id, cam_id, start_time, end_time, type)

        data['records'] = rows
        # 統計物件計數數值
        statistics = {}
        for row in rows:
            key = row['camId']
            type = row['type']
            # 判斷此型態的物件是否存在
            if key not in statistics:
                statistics[key] = {}
                statistics[key] = {}
            if type not in statistics[key]:
                statistics[key][type] = {'inCounter': 0, 'outCounter': 0, 'inAvgSpeed': 0, 'outAvgSpeed': 0}

            # 累加數據
            statistics[key][type]['inCounter'] += row['inCounter']
            statistics[key][type]['outCounter'] += row['outCounter']
            statistics[key][type]['inAvgSpeed'] = (statistics[key][type]['inAvgSpeed'] +
                                                   row['inAvgSpeed']) // 2 if statistics[key][type]['inAvgSpeed'] > 0 else row['inAvgSpeed']
            statistics[key][type]['outAvgSpeed'] = (statistics[key][type]['outAvgSpeed'] +
                                                    row['outAvgSpeed']) // 2 if statistics[key][type]['outAvgSpeed'] > 0 else row['outAvgSpeed']

        data['statistics'] = statistics

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
            db.add_record(record)
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))
    else:
        return {'detail': "OK"}


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
        return {'detail': "OK"}


@app.post("/camera/records")
def get_records(startTime: int, endTime: int, apiUrl: str, EventTypes: str, type: Optional[str] = 'ALL'):
    '''
    查詢某時間段的所有攝影機記錄的資訊\n
    ● startTime(required): 開始時間 (long time to millisecond)\n
    ● endTime(required): 結束時間 (long time to millisecond)\n
    ● apiUrl: NVR的URL路徑\n
    ● EventTypes: 事件類型\n
        LOITERING: 電子圍籬
        STOPPED: 違停
        ENTER: 人車量資訊

    ● type:\n
        ALL: 全部(列出所有分類)
        TRUCK: 大貨車
        PICKUP_TRUCK: 小貨車
        BUS: 公車
        AUTOCAR: 自用車
        MOTORCYCLE: 機車
        BIKE: 腳踏車
        AMBULANCE: 救護車
        FIRE_ENGINE: 消防車
        POLICE_CAR: 警察車
        PEOPLE: 行人
    Response說明:\n
    code: 執行API結果; 0 = 成功, 1 = 失敗\n
    message: 執行結果; 成功=空字串, 失敗=錯誤訊息\n
    data: 回傳所有攝影機統計內容;NVR影像來源ID包含該時段的各類型總數\n
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
        rows = db.get_record(start_time=start_time, end_time=end_time, type=type)

        # 統計物件計數數值
        statistics = {}
        for row in rows:
            key = row['camId']
            type = row['type']
            # 判斷此型態的物件是否存在
            if key not in statistics:
                statistics[key] = {}
                statistics[key] = {}
            if type not in statistics[key]:
                statistics[key][type] = {'inCounter': 0, 'outCounter': 0}

            # 累加數據
            statistics[key][type]['inCounter'] += row['inCounte']
            statistics[key][type]['outCounter'] += row['outCounter']

        # 轉換資料格式符合需求文件
        for key in statistics.keys():
            temp = {'camera': key, 'detail': []}
            for objType in statistics[key].keys():
                temp['detail'].append({
                    'type': objType,
                    'inCounter': statistics[key][objType]['inCounter'],
                    'outCounter': statistics[key][objType]['outCounter']
                })
            data.append(temp)
    except Exception as err:
        return resp(str(err))
    else:
        return resp(None, data)


@app.get("/camera/traffic")
def get_traffic(camera: str, apiUrl: str, endTime: Optional[int] = int(dt.now().timestamp())):
    '''
    查詢即時路況\n
    ● camera: NVR影像來源ID\n
    ● apiUrl: NVR的URL路徑\n

    備註:\n
    traffic:路況類型\n
        JAMMED:壅塞(車速 < 30 & 車輛計數 > 10)
        HEAVY:車多(60 < 車速 > 30 & 10 < 車輛計數 > 3)
        LIGHT:順暢(車速 > 60 & 車輛計數 < 3)
    '''
    # 取得目前時間前五分鐘內的資料,因AI模組每5分鐘更新資料一次
    end_time = dt.fromtimestamp(endTime)
    start_time = end_time - timedelta(minutes=5)
    data = {'camera': camera, 'traffic': "LIGHT"}
    try:
        # 指定type=AUTOCAR
        rows = db.get_record(camId=camera, start_time=start_time, end_time=end_time, type='AUTOCAR')
        carCounter = 0
        carSpeed = 0
        for row in rows:
            if row['type'] != "person":
                carCounter += row['inCounter']
                if row['inAvgSpeed']:
                    carSpeed = (carSpeed + row['inAvgSpeed']) // 2 if carSpeed else row['inAvgSpeed']

        # 根據車速與車輛計數判斷道路狀況, so far we just check trafic base on speed
        if 0 < carSpeed < 30:
            data['traffic'] = "JAMMED"
        elif 30 < carSpeed < 60:
            data['traffic'] = "HEAVY"
        else:
            data['traffic'] = "LIGHT"
    except Exception as err:
        return resp(str(err))
    else:
        return resp(None, data)


@app.get("/camera/statistics/traffic")
def get_statistics_traffic(camera: str, apiUrl: str, startTime: int, endTime: int):
    '''
    查詢小時路況統計清單\n
    ● camera: NVR影像來源ID\n
    ● apiUrl: NVR的URL路徑\n

    備註:\n
    traffic:路況類型\n
        JAMMED:壅塞(車速 < 30 & 車輛計數 > 10)
        HEAVY:車多(60 < 車速 > 30 & 10 < 車輛計數 > 3)
        LIGHT:順暢(車速 > 60 & 車輛計數 < 3)
    '''
    # conver milliseconds to date type
    start_time = dt.fromtimestamp(startTime / 1000.0)
    end_time = dt.fromtimestamp(endTime / 1000.0)
    statisticsList = []
    try:
        # 指定type=AUTOCAR
        rows = db.get_record(camId=camera, start_time=start_time, end_time=end_time, type='AUTOCAR')
        # 從取得的資料中判斷每個小時的道路狀況
        start = start_time
        while True:
            # 設定區間結束時間為區間起始時間+1小時
            end = start + timedelta(hours=1)
            # 如果區間結束時間時間已經超過使用者指定的結束時間,修改區間結束時間設定為使用者指定的結束時間
            if end > end_time:
                end = end_time
            trafficData = {'statsTime': start.timestamp() * 1000, 'camera': camera, 'traffic': "LIGHT"}
            carCounter = 0
            carSpeed = 0
            for row in rows:
                if row['type'] != "person" and row['time'] > start and row['time'] < end:
                    carCounter += row['inCounter']
                    if row['inAvgSpeed']:
                        carSpeed = (carSpeed + row['inAvgSpeed']) // 2 if carSpeed else row['inAvgSpeed']
                    print("Start:{} End:{} Data:{} Speed:{}".format(start, end, row, carSpeed))
            # 根據車速與車輛計數判斷道路狀況, so far we just check trafic base on speed
            if 0 < carSpeed < 30:
                trafficData['traffic'] = "JAMMED"
            elif 30 < carSpeed < 60:
                trafficData['traffic'] = "HEAVY"
            else:
                trafficData['traffic'] = "LIGHT"

            statisticsList.append(trafficData)
            # 執行下一個小時的道路狀況判定
            start = end
            # 已到達使用者指定的結束時間,結束迴圈
            if end == end_time:
                break
    except Exception as err:
        return resp(str(err))
    else:
        # 根據API文件設定回傳值
        return {'code': "0", 'message': "", 'statisticsList': statisticsList}


@app.post("/nvr/modifyPwd")
def update_nvr_config(account: Optional[str] = None, password: Optional[str] = None, deviceUrl: Optional[str] = None, port: Optional[int] = None):
    '''
    更新NVR主機的設定值\n
    ● account: 帳號\n
    ● password: 修改後的密碼 最長為10碼英數字,最少1碼英數字\n
    ● deviceUrl: NVR主機\n
    '''

    config = {}
    try:
        config = db.update_config(account, password, deviceUrl, port)
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))
    else:
        return config


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
