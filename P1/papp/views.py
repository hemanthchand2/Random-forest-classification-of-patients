from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Create your views here.
def home(request):
    data =pd.read_csv(r"C:\Users\heman\Downloads\2023_24projects\2023_projects\37_PatientTreatmentClassification\data.csv")
    X = data[['HAEMATOCRIT', 'HAEMOGLOBINS', 'ERYTHROCYTE', 'LEUCOCYTE', 'THROMBOCYTE', 'MCH', 'MCHC', 'MCV', 'AGE', 'SEX']]
    y = data['SOURCE']  # Assuming 'PATIENT_TYPE' is the column containing the target labels ('in' or 'out')
    y = y.map({'out': 0, 'in': 1})
    X['SEX'] = X['SEX'].map({'M': 0, 'F': 1})
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=50)
    rf_classifier.fit(X_train, y_train)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    result=''
    if(request.method=="POST"):
        data1=request.POST
        HAEMATOCRIT=data1.get('HAEMATOCRIT')
        HAEMOGLOBINS=data1.get('HAEMOGLOBINS')
        ERYTHROCYTE=data1.get('ERYTHROCYTE')
        LEUCOCYTE=data1.get('LEUCOCYTE')
        THROMBOCYTE=data1.get('THROMBOCYTE')
        MCH=data1.get('MCH')
        MCHC=data1.get('MCHC')
        MCV=data1.get('MCV')
        AGE=data1.get('AGE')
        SEX=data1.get('SEX')

        if not all([HAEMATOCRIT, HAEMOGLOBINS, ERYTHROCYTE, LEUCOCYTE, THROMBOCYTE, MCH, MCHC, MCV, AGE, SEX]):
            result = "All fields are mandatory for the result to be accurate"
            return render(request, "index.html", context={'result': result})

        if('submit' in request.POST):
            if (SEX=='M' or 'm'):
                SEX=0
            elif(SEX=='F'or 'f'):
                SEX=1
            else:
                result="Please enter the SEX as M/m for male or F/f for female "
                return render(request, "index.html", context={'result': result})
            op = rf_classifier.predict([[float(HAEMATOCRIT),float(HAEMOGLOBINS),float(ERYTHROCYTE),float(LEUCOCYTE),float(THROMBOCYTE),float(MCH),float(MCHC),float(MCV),float(AGE),float(SEX)]])
            if(op==1):
                result="IN-PATIENT"
            else:
                result="OUT-PATIENT"
    return render(request,"index.html",context={'result':result})