# Forecast No. of Orders TalaadThai intership 2024
This project is Machine Learning Project, forecasting number of orders TalaadThai. The idea is use time series model to forecast/future predict number of orders. The models are Regression Models, Statistic Models and Deep Learning Models Create By Mr. Polakorn Anantapakorn MUICT student year 2.  

## Datasets
### Provided dataset 
เราใช้ข้อมูลเกี่ยวกับรายการคำสั่งซื้อ โดยข้อมูล Query จาก PostgresSQL Database ข้อมูล Dataset จะอยู่ภายใน Folder "csv" ไฟล์ข้อมูลจะเป็นรูปแบบ CSV ที่ประกอบไปด้วยคอลัมน์ order_completed_at แสดงถึงข้อมูลวันเดือนปี และคอลัมน์ unique_order_count แสดงถึงจำนวนรายการสินค้าในวันเดือนปีนั้น ช่วงวันที่ข้อมูล datatset 2023-01-01 ถึง 2024-07-02 (ปี-เดือน-วัน)

## Project Structure

```

D:.
├───deepLearning_model
│   ├───best_check
│   └───weights_check
└───timeSeries_model
    ├───csv
    │   └───total_orderJun67
    ├───forecast
    │   ├───predict-19Jun67
    │   ├───predict-20Jun67
    │   ├───predict-21Jun67
    │   ├───predict-25Jun67
    │   ├───predict-2Jul67
    │   └───usePredict
    ├───model
    ├───model_old
    │   ├───catboost
    │   ├───exponentialSmooth
    │   ├───lightGBM
    │   ├───linearReg
    │   ├───model_25Jun67
    │   ├───prophet
    │   └───randomForest
    ├───notebook
    └───notebook_old

```

Folder Project ประกอบไปด้วย Folder Time Series Model ที่จะเป็นไปด้วย Statistic Model และ Regression Model และ Dataset ที่ใช้งาน ถัดมาเป็น Deep Learning Model ที่จัดเก็บ Deep Learning Model

### ประเภทไฟล์เบื้องต้น
1. ไฟล์ที่คำขึ้นต้นด้วย train_.ipynb หรือ demo_.ipynb จะเป็นประเภทไฟล์พัฒนา ML(Mahchine Learning) Model และทดสอบ Model รวมถึงการเรียก Model ทำนายผลลัพท์
2. ไฟล์ที่ลงท้ายนามสกุล .keras เป็น Model ที่พัฒนาบันทึกได้จัดเก็บไว้ภายใน Folder ต่างๆ (Folder Model และ Model_old) สามารถเรียกใช้เพื่อนำไปพัฒนาต่อยอด หรือ เรียกใช้นำมาทำนายผลได้
3. ไฟล์ประเภท csv ที่อาจจะเป็นไฟล์ Dataset หรือ ไฟล์ Forecast ที่โมเดลทำนายผลลัพท์ออกมา (Folder csv จัดเก็บ Dataset, Folder forecast จัดเก็บผลทำนาย)

## Usage
### Requirements:

Project ที่จัดทำขึ้นนี้ Folder timeSeries_model และ deepLearning_model ใช้งาน Library ที่แตกต่างกันมีรายละเอียดดังนี้

หมายเหตุ ผู้จัดทำขอข้ามขั้นตอนการสร้าง Environment เพราะรายละเอียดของทั้งสอง Folder มีขั้นตอนและวิธีการแตกต่างกัน ผู้จัดทำจึงให้ช่องทางการสร้าง Environment ดังนี้

Folder timeSeries_model พัฒนาภายใน Anaconda Environment (conda environment)
https://www.youtube.com/watch?v=fnb4_MzpZFU

Folder deepLearning_model พัฒนาภายใน venv Environment
https://code.visualstudio.com/docs/python/environments#:~:text=To%20create%20local%20environments%20in,environment%20types%3A%20Venv%20or%20Conda.

#### timeSeries_model Important Libraries:
python 3.12.3
```
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.0
seaborn==0.13.2
statsmodels==0.14.2
u8darts==0.29.0
u8darts-all==0.29.0
matplotlib-base==3.8.4
```

#### deepLearning_model Important Libraries:
python 3.10.11
```
keras==3.4.1
matplotlib==3.8.3
numpy==1.26.4
pandas==2.2.1
scikit-learn==1.4.1.post1
tensorflow==2.16.2
tensorflow-cpu==2.16.2
```


## วิธีการรันโปรเจ็ค 
การพัฒนาไฟล์โปรเจ็คพัฒนาอยู่บนไฟล์ .ipynb (่jupyter notebook) Programming language ที่ใช้งานคือ Python
ไฟล์ที่พัฒนาส่วนใหญ่ประกอบไฟด้วย Topic ต่างๆ ตามกระบวนการพัฒนา Machine Learning Model

## Machine Learning Process
ผู้จัดทำจะอธิบายขั้นตอนการพัฒนาโมเดลหัวข้อต่างๆ ( timeSerie Model [Statistic Model, Regression Model]  และ Deep Learning ตามลำดับ ) ที่ผู้จัดทำได้ดำเนินมามีรายละเอียดดังนี้

1. Import Data
2. Data Preprocessing and Feature Engineering
3. Modeling and Training Model
4. Evaluation
5. Result and Summary


### การนำเข้าข้อมูล (Import Data)
การนำเข้าข้อมูลเพื่อให้ Machine Learning เรียนรู้และทำนายผลลัพท์ ตัวอย่างการเรียกข้อมูล

![import_data](picture_markdown/import_data.png)

### การวิเคราะห์ข้อมูลและการแปลงข้อมูล (Data Preprocessing)
ยกตัวอย่างเทคนิค Data Preprocessing ที่นำมาปรับใช้ภายในโปรเจ็คนี้

#### การเปลี่ยน Datatype:
การเปลี่ยน dataframe ซึ่งเป็นข้อมูลที่ได้จากการนำเข้าข้อมูล เปลี่ยนเป็น Time Series Object เนื่องจากการใช้งาน Darts Library จึงต้องเปลี่ยนข้อมูลให้เหมาะสม

![timeSeriesObject](picture_markdown/timeSeriesObject.png)

#### การลบค่า Outiler (ค่าผิดปกติ):
การลบค่าผิดปกติของข้อมูลช่วยให้ Machine Learning เรียนรู้ได้เหมาะสมมากขึ้นและเรียนรู้ Pattern ของข้อมูลที่เหมาะสม
ตัวอย่างการนำค่าผิดปกติแทนค่าตัวค่ากลาง

***before remove outlier***
![before_outlier](picture_markdown/before_outlier.png)

***after remove outlier***
![after_outlier](picture_markdown/remove_outiler.png)

#### Feature Scaling:
การปรับค่าข้อมูลให้มีขนาดช่วงระหว่าง 0 ถึง 1 เพื่อให้ Machine Learning และ Deep Learning เรียนรู้ได้ดียิ่งขึ้น

![feature_scaling](picture_markdown/remove_outiler.png)

### Modeling and Training Model
การออกแบบโมเดลและพัฒนาโมเดล ให้มีความเหมาะสมกับข้อมูลที่เรียนรู้และการนำไปใช้งาน 

#### การแบ่งข้อมูลชุดเรียนรู้และข้อมูลชุดทดสอบ (Train Test Split)
การแบ่งข้อมูลที่ให้ Machine Learning เรียนรู้และข้อมูลที่ใช้สำหรับทดสอบ
ตัวอย่างการแบ่ง Train Test Split
![train_test_split](picture_markdown/train_test_split.png)

#### Statistic Model และ Regression Model
ภาพตัวอย่าง การ train model ของ Linear Regression ที่กำหนดให้ Model ท้ายผล 1 สัปดาห์ โดยข้อมูล
![Train_linear_Regression](picture_markdown/linearModel.png)

#### GridSearch 
gridSearch คือ function ช่วยเหลือนักพัฒนาโมเดล ให้ผลลัพท์ Model ที่เหมาะสมที่สุด จากการทดลองกับ parameter ต่างๆของโมเดล เป็นวิธีการ Tuning Model อย่างหนึ่ง

![gridsearch](picture_markdown/gridsearch.png)

### Evaluation
การทดสอบประสิทธิภาพโมเดล

#### Performance Metric
ค่าดัชนีที่ใช้วัดผลของ Machine Learning มีรายละเอียดเบื้องต้นดังนี้:

1. MAE, MSE ,RMSE เป็นค่าดัชนีที่ใช้วัดค่าความคลาดเคลื่อนมีสูตรการคำนวณที่แตกต่างกัน แต่โดยรวมมีจุดประสงค์ให้ผู้พัฒนาสามารถเห็นค่าเคลื่อนคลาด

2. MAPE เป็นค่าดัชนีใช้วัดค่าความคลาดเคลื่อนค่าเป็นร้อยละ (%) และ Accuracy เป็นค่าส่วนกลับของ MAPE (Accuracy = 100 - MAPE)

![metric](picture_markdown/metric.png)


#### Evaluation Train Test Split
การทดสอบโมเดลกับข้อมูลชุดข้อมูลและนำมาเปรียบเทียบ ตัวอย่างการทดสอบ Linear Regression 
![prediction_plot](picture_markdown/pred_plot.png)

### Result and Summary
การสรุปผลโมเดลที่ได้ทดสอบ จากการทดสอบโมเดลได้ผลลัพท์โปรเจ็คในระยะแรก ผู้พัฒนา