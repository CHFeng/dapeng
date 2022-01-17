## create virtual enviroment
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