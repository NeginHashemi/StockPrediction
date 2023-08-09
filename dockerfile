from python:3.8

RUN pip install torch==2.0.1+cpu torchvision==0.15.2+cpu -f https://download.pytorch.org/whl/torch_stable.html

Run pip install asyncio==3.4.3 uvicorn==0.22.0 Pillow==9.4.0 fastapi==0.96.0 mlflow==2.4.1 psutil==5.9.4 Flask-Cors==4.0.0

ENV DIR /opt/image-service

RUN mkdir $DIR

WORKDIR $DIR

COPY requirements.txt $DIR/

RUN pip install -r requirements.txt

COPY app.py $DIR/
COPY metrics.py $DIR/
COPY stock_load_model.py $DIR/

COPY index.html $DIR/

COPY vabemellat_30min_200.csv $DIR/

COPY modelV2.pth $DIR/

COPY modelV1.pth $DIR/

CMD python app.py


