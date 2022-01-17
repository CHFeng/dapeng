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
啟動服務
```
uvicorn app:app --reload
```