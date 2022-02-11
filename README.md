## create virtual enviroment
[參考](https://fastapi.tiangolo.com/zh/)

利用conda建立虛擬環境
```
conda create --name dapeng python==3.7
conda activate dapeng
```
## FastAPI
安裝套件
```
pip install fastapi
pip install uvicorn
```
啟動服務,reload功能可讓服務有變更程式碼時重新讀取
```
uvicorn app:app --reload
```
正確啟動後，可以透過http://127.0.0.1:8000/docs#/ 查看API文件

## PostgreSQL
[參考](https://medium.com/alberthg-docker-notes/docker%E7%AD%86%E8%A8%98-%E9%80%B2%E5%85%A5container-%E5%BB%BA%E7%AB%8B%E4%B8%A6%E6%93%8D%E4%BD%9C-postgresql-container-d221ba39aaec)

透過docker安裝postgres
```
# 先 pull image
$ docker pull postgres
# 指定port number & password來啟動postgres
$ docker run -d --name my-postgres -p 8080:5432 -e POSTGRES_PASSWORD=admin postgres
```
進入 PostgreSQL 的 CLI 命令列介面
這裡會使用 exec 的 -i 與 -t 參數，讓終端機保持開啟。當進入 PostgreSQL 後若想離開，則可輸入 「\q」即可。
```
$ docker exec -it my-postgres psql -U postgres
```
建立使用者與新增資料庫,其中的user-name與password是可以自行修改的
```
docker exec -it my-postgres psql -U postgres -c "create role <user-name> with login password '<password>';"
```
建立資料庫，其中的 database-name 與 user-name是可以自行修改的
```
$ docker exec -it my-postgres psql -U postgres -c "create database <database-name> owner <user-name>"
```
檢查資料庫列表：
```
$ docker exec -it my-postgres psql -U postgres -c "\l"
```
安裝python library
```
pip install psycopg2
```

## Pyinstaller
[參考](https://zh-tw.coderbridge.com/@WeiHaoEric/0b2ced0696cc4c38a62d7b26fa7bbea0)

使用pyinstaller將source code編譯成可執行檔

安裝library
```
pip install pyinstaller
```
編譯
```
pyinstaller -F -n 編譯後的檔名 來源檔名
```
