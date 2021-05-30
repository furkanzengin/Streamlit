import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import altair as alt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler
from streamlit.proto.Slider_pb2 import Slider
import SessionState
import login
import train
from sklearn.model_selection import train_test_split
import base64,pickle
def download_link(object_to_download, download_filename, download_link_text):
    """
    Generates a link to download the given object_to_download.

    object_to_download (str, pd.DataFrame):  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv, some_txt_output.txt
    download_link_text (str): Text to display for download link.

    Examples:
    download_link(YOUR_DF, 'YOUR_DF.csv', 'Click here to download data!')
    download_link(YOUR_STRING, 'YOUR_STRING.txt', 'Click here to download your text!')

    """
    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

def download_model(object_to_download, download_filename, download_link_text):
    output_model = pickle.dumps(object_to_download)
    b64 = base64.b64encode(output_model).decode()
    return f'<a href="data:file/output_model;base64,{b64}" download="{download_filename}">{download_link_text}</a>'
    
def getSess():
  return SessionState.get(ml="",x=pd.DataFrame(), prev=pd.DataFrame(), filterDf=pd.DataFrame(), dep=[], ind=[], filterList = [], loggedIn = False, control = True)

def dataframe(df):
  st.dataframe(df.sample(20, random_state=42))

def title(isim):
  return st.sidebar.header(isim)


def visualization():
  title("GÖRSELLEŞTİRME")
  
  st.sidebar.write("Eğer veri setinizi belirli bir değer aralığına göre sınırlamak istiyorsanız lütfen değerleri giriniz")
  st.sidebar.header("Sayısal veriler için")
  cols = st.sidebar.selectbox("Sınırlamak istediğiniz değişkeni seçin", ss.filterDf.columns)
  startValue = st.sidebar.number_input("Başlangıç Değeri", value=0)
  endValue = st.sidebar.number_input("Son Değeri",value=0)

  filtre = st.sidebar.button("Sınırlandır")
  
  if filtre:
      ss.filterDf = ss.filterDf.loc[(ss.filterDf[cols] >= startValue) & (ss.filterDf[cols] <= endValue)]
      ss.filterList.append(cols+ ", min: " + str(ss.filterDf[cols].min()) + ", max: "+ str(ss.filterDf[cols].max()))
      for i in ss.filterList:
        st.sidebar.write(i)
  
  st.sidebar.header("Kategorik Veriler İçin")

  col = st.sidebar.selectbox("Sınırlamak istediğiniz sütunu seçin", ss.x.columns)
  cols = st.sidebar.multiselect("Değerleri seçiniz",ss.x[col].unique())
  filtre = st.sidebar.button("Sınırlandırın")
  

  if filtre:
    if cols != []:
      ss.filterDf = ss.filterDf.loc[ss.filterDf[col].isin(cols)]

  st.sidebar.write("Görselleştirecek satır aralığını sınırla")
  start = st.sidebar.slider("Başlangıç Değer",0,len(ss.x))
  end = st.sidebar.slider("Son Değer",0,len(ss.x),len(ss.x))
  cols = st.sidebar.multiselect("Görselleştirilecek değişkenleri seçin", ss.x.columns)

  
  col1, col2 = st.sidebar.beta_columns(2)
  X = col1.selectbox("X değerini seçin", ss.x.columns)
  Y = col2.selectbox("Y değerini seçin", ss.x.columns)

  col1, col2 = st.sidebar.beta_columns(2)
  hist = col1.button("Göster")
  gizle = col2.button("Gizle")
  st.sidebar.write("Belirlediğiniz Sınır Değerlerini Kaldırabilirsiniz")
  sifirla = st.sidebar.button("Sınırları Kaldır")
  if sifirla:
    ss.filterDf = ss.x
    ss.filterList = []
  if hist:
    st.header("GRAFİKLER")
    st.bar_chart(ss.filterDf[cols][start:end])
    st.line_chart(ss.filterDf[cols][start:end])
    st.area_chart(ss.filterDf[cols][start:end])
    brush = alt.selection_interval()

    chart = alt.Chart(ss.filterDf).mark_point(opacity=1).encode(
      x=X, y = Y).add_selection(brush)

    st.altair_chart(chart,use_container_width=True)

  if gizle:
    pass

def builtSideBar(df):
  ss = getSess()

  geriAl = st.sidebar.button("VERİ SETİ ÜZERİNDEKİ SON İŞLEMİ GERİ AL")
  if geriAl:
    ss.x = ss.prev
  columns = list(ss.x.columns)
  title("SÜTUN AD DEĞİŞTİRİN")
  cols = st.sidebar.selectbox("Adını değiştireceğiniz sütunları seç", ss.x.columns)
  user_input = st.sidebar.text_input("Yeni ad", " ")
  changeName = st.sidebar.button("Adı Değiştir")
  if changeName:
    ss.prev = ss.x.copy()
    ss.x = ss.x.rename(columns={cols:user_input})
    

  title("SÜTUN TİPİNİ DEĞİŞTİRİN")
  cols = st.sidebar.multiselect("Tipini değiştireceğiniz sütunları seç", ss.x.columns)
  tip = st.sidebar.selectbox(
    "Dönüştürebileceğiniz sütun tipleri",
    ("int", "float", "string")
  )
  changeType = st.sidebar.button("Değiştir")
  if changeType:
    ss.prev = ss.x.copy()
    for i in cols:
      ss.x[i] = ss.x[i].astype(tip)

  title("KAYIP VERİLER")
  cols = st.sidebar.multiselect("İşlem yapılacak sütunları seç", options = ss.x.columns)
  col1, col2, col3 = st.sidebar.beta_columns(3)
  mod = col1.button("Mod Deger İle Doldur")
  medyan = col1.button("Medyan Değer İle Doldur")
  ortalama = col2.button("Ortalama ile Doldur")
  minDeger = col2.button("Minimum Değer ile Doldur")
  maxDeger = col3.button("Maximum Değer İle Doldur")
  removeRow = col3.button("Satırları sil")

  if mod:
    for i in cols:
      ss.x[i].fillna(ss.x[i].mode()[0], inplace = True)

    
  if medyan:
    for i in cols:
      ss.x[i].fillna(ss.x[i].median(), inplace = True)
    
  if ortalama:
    for i in cols:
      ss.x[i].fillna(ss.x[i].mean(), inplace = True)
    
  if minDeger:
    for i in cols:
      ss.x[i].fillna(ss.x[i].min(), inplace = True)
    
  if maxDeger:
    for i in cols:
      ss.x[i].fillna(ss.x[i].max(), inplace = True)
    
  if removeRow:
    ss.x = ss.x.dropna()
    
  
  title("Sütun Drop")
  dropList = st.sidebar.multiselect("DROPLANACAK SÜTUNLARI SEÇİNİZ" ,options = ss.x.columns)
  dropButton = st.sidebar.button("DROP")
  if dropButton:
    ss.prev = ss.x.copy()
    for i in dropList:
      ss.x = ss.x.drop(i,axis=1)
    

  title('Kategorik verileri kodlama:')
  kodLE = st.sidebar.multiselect('Kodlanacakk sütunları seçin', ss.x.columns)
  col1, col2 = st.sidebar.beta_columns(2)
  labelEncode = col1.button('LABEL ENCODE')
  ohe = col2.button("ONE HOT ENCODER")
  if labelEncode:
    ss.prev = ss.x.copy()
    for i in kodLE:
      labelEnc = LabelEncoder()
      l_e = labelEnc.fit_transform(pd.DataFrame(ss.x[i]))
      ss.x = ss.x.drop(i,axis=1)
      ss.x.insert(columns.index(i),i, pd.Series(l_e))
      ss = getSess()
  
  if ohe:
    ss.prev = ss.x.copy()
    for col in kodLE:
      enc = OneHotEncoder(sparse=False)
      X = ss.x[col]
      X = X.values.reshape(len(X), 1)
      x_transformed = enc.fit_transform(X)
      columnsOhe = [col+"_"+str(i) for i in ss.x[col].unique()]
      x_ohe = pd.DataFrame(data = x_transformed, columns=columnsOhe)
      ss.x = pd.concat([ss.x,x_ohe] ,axis = 1)

  title("ÖLÇEKLENDİRME")
  col1, col2 = st.sidebar.beta_columns(2)
  min_max = col1.button("MIN-MAX SCALER")
  standart = col2.button("STANDART SCALER")
  if min_max:
    ss.prev = ss.x.copy()
    min_max_scaler = MinMaxScaler()
    columns = ss.x.columns
    x_scaled = min_max_scaler.fit_transform(ss.x)
    ss.x = pd.DataFrame(x_scaled, columns = columns)
    
  if standart:
    ss.prev = ss.x.copy()
    standart_scaler = StandardScaler()
    columns = ss.x.columns
    x_scaled = standart_scaler.fit_transform(ss.x)
    ss.x = pd.DataFrame(x_scaled, columns=columns)

  title("AYKIRI DEĞERLERİ GİDERME")
  outliers = st.sidebar.multiselect('İşlem Yapılmasını istediğiniz sütunları seçip, ardından yapmak istediğiniz işlemi seçin.'
                                  , ss.x.columns)
  col1,col2 = st.sidebar.beta_columns(2)
  removeOutliers = col1.button("SİL")
  baskila = col2.button("BASKILA")
  fillMean = st.sidebar.button("ORTALAMA İLE DOLDUR")

  if removeOutliers:
    ss.prev = ss.x.copy()
    for outlier in outliers:
      df_acc = pd.DataFrame(ss.x[outlier])

      Q1 = df_acc.quantile(0.25)
      Q3 = df_acc.quantile(0.75)
      IQR = Q3 - Q1

      alt_sinir = Q1- 1.5*IQR
      ust_sinir = Q3 + 1.5*IQR

      ss.x = ss.x[~(ss.x[(df_acc < alt_sinir) | (df_acc > (ust_sinir))]).any(axis=1)]

  if baskila:
    ss.prev = ss.x.copy()
    for outlier in outliers:
      df_acc = ss.x[outlier]

      Q1 = df_acc.quantile(0.25)
      Q3 = df_acc.quantile(0.75)
      IQR = Q3 - Q1

      alt_sinir = Q1- 1.5*IQR
      ust_sinir = Q3 + 1.5*IQR
      alt_aykirilar = df_acc < alt_sinir
      ust_aykirilar = df_acc > ust_sinir
      df_acc[alt_aykirilar] = alt_sinir
      df_acc[ust_aykirilar] = ust_sinir
      
  if fillMean:
    ss.prev = ss.x.copy()
    for outlier in outliers:
      df_acc = ss.x[outlier]

      Q1 = df_acc.quantile(0.25)
      Q3 = df_acc.quantile(0.75)
      IQR = Q3 - Q1

      alt_sinir = Q1- 1.5*IQR
      ust_sinir = Q3 + 1.5*IQR
      tum_aykirilar = (df_acc < (alt_sinir)) | (df_acc > (ust_sinir) )
      df_acc[tum_aykirilar] = df_acc.mean()

  title("BAĞIMLI BAĞIMSIZ DEĞİŞKENLERİ BELİRLEME")
  variables = st.sidebar.multiselect('Değişkenleri seçin', ss.x.columns)
  col1, col2 = st.sidebar.beta_columns(2)
  bagimsiz = col1.button("Bağımsız Değişken Olarak Belirle")
  bagimli = col2.button("Bağımlı Değişken Olarak Belirle")
  sifirla = st.sidebar.button("Bağımlı-Bağımsız Değişkenleri Sıfırla")

  if bagimsiz:
    ss.ind = variables
  if bagimli:
    ss.dep = variables
  if sifirla:
    ss.ind, ss.dep = [],[]
  



  dataframe(ss.x)

if __name__ == '__main__':
  ss = getSess()
  anasayfa = st.button("Ana Sayfa")
  if (ss.loggedIn == False) or (ss.loggedIn == None):
    if anasayfa:
      st.error("Anasayfaya Gitmek İçin Giriş Yapınız!")
    a = login.loginPage()
    if a:
      ss.loggedIn = a

  elif ss.loggedIn == True:
    uploaded_file = st.file_uploader("Veri Setinizi Yükleyiniz.")
    if uploaded_file is None:
      ss.control = True
    if uploaded_file is not None:
      df = pd.read_csv(uploaded_file, parse_dates=True)
      if ss.control:
        ss.x = df
        ss.filterDf = df
        ss.prev = df
        ss.control = False
        ss.ind, ss.dep = [],[]
      st.header("ORİJİNAL VERİLER")
      dataframe(df)
      st.header("İŞLENMİŞ VERİLER")
      builtSideBar(df)
      col1, col2 = st.beta_columns(2)
      col1.header("VERI SETI SUTUN TANIMLAYICI BILGILERI")
      col1.write(ss.x.describe())
      dic = {}
      for col in ss.x:
        dic[col] = [ss.x[col].isna().sum()]
      col2.header("KAYIP DEGERLERIN TOPLAMI")
      col2.write(pd.DataFrame(data=dic))
      col2.header("SÜTUN TİPLERİ")
      dic = {}
      for col in ss.x:
        dic[col] = [ss.x[col].dtypes]
      col2.write(pd.DataFrame(data=dic))

      st.title("Aykırı Değerler")
      dic = {}
      for col in ss.x:
        if ss.x[col].dtype == 'int64' or ss.x[col].dtype == 'float64' or ss.x[col].dtype == 'int32' or ss.x[col].dtype == 'float32' :
          Q1 = ss.x[col].quantile(0.25)
          Q3 = ss.x[col].quantile(0.75)
          IQR = Q3 - Q1

          lower_bound = Q1- 1.5*IQR
          upper_bound = Q3 + 1.5*IQR
          toplam = len(ss.x.loc[( ss.x[col] < (lower_bound)) | ( ss.x[col] > (upper_bound))])
          dic[col] = {'q1':Q1,'q3':Q3,'iQR':IQR,'alt sinir':lower_bound,'Ust sinir':upper_bound,
          'toplam Aykiri':toplam}
      st.write(pd.DataFrame(data=dic))

      st.header("Bağımlı Bağımsız Değişkenler")
      if ss.dep==[] and ss.ind==[]:
        st.write("Henüz Değişkenleri Belirlemediniz!")
      col1, col2 = st.beta_columns(2)
      col1.write(pd.Series(ss.ind, name="Bagimsiz Degiskenler"))
      col2.write(pd.Series(ss.dep, name="Bagimli Degiskenler"))
      
      visualization()

      title("MAKİNE ÖĞRENMESİ ALGORİTMALARI")

      regOrClass = st.sidebar.selectbox("Algoritma Türünü Seçiniz", ("Regresyon", "Sınıflandırma"))
      if regOrClass == "Regresyon":
        model = st.sidebar.selectbox("Algoritmalar",
                              ("Linear Regression","SVR", "Decision Tree Regressor", "Random Forest"))
      else:
        model = st.sidebar.selectbox("Algoritmalar",
                              ("Logistic Regression","KNN", "SVC", "Naive Bayes", "Decision Tree Classifier"))
        
      slider = st.sidebar.slider("Train-Test Split Oranını Belirleyin(Yüzdelik)",min_value=1,max_value=99,
                                  value=33)
      training = st.sidebar.button("Train")
      
      if training:
        dic = {}
        st.header("Test Verilerine Göre Model Başarı Sonuçları")
        X_train, X_test, y_train, y_test = train_test_split(ss.x[ss.ind],ss.x[ss.dep], test_size=slider/100, random_state=42)
        if model == "Linear Regression":
            ss.ml, mae, mse, r2 = train.linearRegr(X_train, X_test, y_train, y_test)
            dic["Model"] = ["Linear Regression"]
            dic["Mean Absolute Error"] = [mae]
            dic["Mean Squared Error"] = [mse]
            dic["R2 Score"] = [r2]
            st.write(pd.DataFrame(data=dic))
        elif model == "SVR":
            ss.ml, mae, mse, r2 = train.svr(X_train, X_test, y_train, y_test)
            dic["Model"] = ["SVR"]
            dic["Mean Absolute Error"] = [mae]
            dic["Mean Squared Error"] = [mse]
            dic["R2 Score"] = [r2]
            st.write(pd.DataFrame(data=dic))

        elif model == "Decision Tree Regressor":
            ss.ml, mae, mse, r2 = train.decisionTreeRegr(X_train, X_test, y_train, y_test)
            dic["Model"] = ["Decision Tree Regressor"]
            dic["Mean Absolute Error"] = [mae]
            dic["Mean Squared Error"] = [mse]
            dic["R2 Score"] = [r2]
            st.write(pd.DataFrame(data=dic))

        elif model == "Random Forest":
            ss.ml, mae, mse, r2 = train.randomForestRegr(X_train, X_test, y_train, y_test)
            dic["Model"] = ["Random Forest"]
            dic["Mean Absolute Error"] = [mae]
            dic["Mean Squared Error"] = [mse]
            dic["R2 Score"] = [r2]
            st.write(pd.DataFrame(data=dic))

        elif model == "Logistic Regression":
            ss.ml, mae, acc, avp, conf, f1, recall, prec = train.logisticRegress(X_train, X_test, y_train, y_test)
            dic["Model"] = ["Logistic Regression"]
            dic["Mean Absolute Error"] = [mae]
            dic["Accuracy"] = [acc]
            dic["Average Precision Score"] = [avp]
            dic["F1 Score"] = [f1]
            dic["Recall"] = [recall]
            dic["Precision"] = [prec]
            st.write(pd.DataFrame(data=dic))
            st.header("Confusion Matrix")
            st.write(conf)
        elif model == "KNN":
            ss.ml, mae, acc, avp, conf, f1, recall, prec = train.kNN(X_train, X_test, y_train, y_test)
            dic["Model"] = ["KNN"]
            dic["Mean Absolute Error"] = [mae]
            dic["Accuracy"] = [acc]
            dic["Average Precision Score"] = [avp]
            dic["F1 Score"] = [f1]
            dic["Recall"] = [recall]
            dic["Precision"] = [prec]
            st.write(pd.DataFrame(data=dic))
            st.header("Confusion Matrix")
            st.write(conf)
        elif model == "SVC":
            ss.ml, mae, acc, avp, conf, f1, recall, prec = train.svc(X_train, X_test, y_train, y_test)
            dic["Model"] = ["SVC"]
            dic["Mean Absolute Error"] = [mae]
            dic["Accuracy"] = [acc]
            dic["Average Precision Score"] = [avp]
            dic["F1 Score"] = [f1]
            dic["Recall"] = [recall]
            dic["Precision"] = [prec]
            st.write(pd.DataFrame(data=dic))
            st.header("Confusion Matrix")
            st.write(conf)
        elif model == "Decision Tree Classifier":
            ss.ml, mae, acc, avp, conf, f1, recall, prec = train.decisionTreeClass(X_train, X_test, y_train, y_test)
            dic["Model"] = ["Decision Tree Classifier"]
            dic["Mean Absolute Error"] = [mae]
            dic["Accuracy"] = [acc]
            dic["Average Precision Score"] = [avp]
            dic["F1 Score"] = [f1]
            dic["Recall"] = [recall]
            dic["Precision"] = [prec]
            st.write(pd.DataFrame(data=dic))
            st.header("Confusion Matrix")
            st.write(conf)
        elif model == "Naive Bayes":
            ss.ml, mae, acc, avp, conf, f1, recall, prec = train.GaussianNaive(X_train, X_test, y_train, y_test)
            dic["Model"] = ["Naive Bayes"]
            dic["Mean Absolute Error"] = [mae]
            dic["Accuracy"] = [acc]
            dic["Average Precision Score"] = [avp]
            dic["F1 Score"] = [f1]
            dic["Recall"] = [recall]
            dic["Precision"] = [prec]
            st.write(pd.DataFrame(data=dic))
            st.header("Confusion Matrix")
            st.write(conf)

     
      title("İNDİRMELER")

      st.sidebar.header("İşlenmiş veriler")
      download_csv = st.sidebar.button("CSV Olarak İndirin")
      if download_csv:
          tmp_download_link = download_link(ss.x, "new_data.csv", 'CSV Dosyanız Hazır. İndirmek için tıklayın.')
          st.sidebar.markdown(tmp_download_link, unsafe_allow_html=True)
      st.sidebar.header("Eğitilmiş model")
      if ss.ml != "":
        download_ml = st.sidebar.button("Modelinizi İndirin")
        if download_ml:
            tmp_download_link = download_model(ss.ml, model+"_model.pkl", 'Model Dosyanız Hazır. İndirmek için tıklayın.')
            st.sidebar.markdown(tmp_download_link, unsafe_allow_html=True)
            