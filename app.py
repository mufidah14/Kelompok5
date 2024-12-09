import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os

# Fungsi untuk memuat data
@st.cache_data
def load_data():
    data = pd.read_csv('diabetes.csv')
    return data

# Fungsi untuk memuat atau melatih model Logistic Regression
@st.cache_resource
def load_model():
    model_path = 'model/diabetes_logistic_model.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
    else:
        # Jika model tidak ada, latih model Logistic Regression
        data = load_data()
        X = data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
        y = data['Outcome']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        
        # Evaluasi akurasi model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Akurasi model Logistic Regression: {accuracy:.4f}")
        
        # Simpan model ke file
        os.makedirs('model', exist_ok=True)
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)
    
    return model

# Sidebar Menu
menu = st.sidebar.selectbox(
    "Pilih Menu",
    options=["Home", "Tampilkan Dataset", "Visualisasi Data", "Prediksi Diabetes"]
)

# Menu: Home
if menu == "Home":
    st.title("Prediksi Diabetes Menggunakan Logistic Regression")
    st.image(
        "kesehatan diabetes.jpg", 
        caption="Ilustrasi Prediksi Diabetes",
        use_container_width=True
    )
    st.markdown("<h2 style='text-align: center; font-weight: bold;'>Selamat datang di Aplikasi Prediksi Diabetes!</h2>", unsafe_allow_html=True)
    st.markdown("""
    Aplikasi ini dirancang untuk membantu Anda memahami risiko diabetes berdasarkan data kesehatan pribadi Anda. 
    Dengan menggunakan teknik Logistic Regression, aplikasi ini mampu memproses berbagai data kesehatan dan memberikan 
    prediksi apakah Anda berisiko mengalami diabetes atau tidak.
    """)

# Menu: Tampilkan Dataset
elif menu == "Tampilkan Dataset":
    st.title("Dataset Diabetes")
    data = load_data()
    st.write("Berikut adalah dataset yang digunakan dalam aplikasi ini:")
    st.dataframe(data)
    st.markdown("""
    Pada Dataset aplikasi prediksi diabetes di atas berisi data pasien yang digunakan untuk memprediksi kemungkinan terdiagnosis diabetes. 
    Berikut adalah penjelasan dari setiap kolom dalam dataset:
    """)

    st.markdown("""
    1. **Pregnancies** : Menunjukkan jumlah kehamilan yang dialami oleh seorang wanita.
    2. **Glucose**: Tingkat glukosa dalam darah, diukur dalam satuan mg/dL. Glukosa tinggi dapat menjadi indikator risiko diabetes.
    3. **BloodPressure**: Tekanan darah diastolik (mm Hg), menunjukkan tekanan darah dalam arteri antara detak jantung.
    4. **SkinThickness**: Ketebalan lipatan kulit (mm), digunakan sebagai indikasi tingkat lemak tubuh.
    5. **Insulin**: Tingkat insulin serum (IU/mL), memberikan indikasi tentang sensitivitas insulin pasien.
    6. **BMI**: Indeks Massa Tubuh (Body Mass Index), dihitung berdasarkan berat badan dan tinggi badan, memberikan indikasi status berat badan.
    7. **DiabetesPedigreeFunction**: Nilai fungsi diabetes keturunan, menunjukkan risiko genetik berdasarkan riwayat keluarga.
    8. **Age**: Usia pasien (tahun).
    9. **Outcome**: Hasil diagnostik, di mana 0 menunjukkan tidak terdiagnosis diabetes, dan 1 menunjukkan pasien terdiagnosis diabetes.
    """)

    st.write(f"**Total Data:** {data.shape[0]} baris dan {data.shape[1]} kolom.")

    st.subheader("Statistik Deskriptif")
    st.write(data.describe())
    st.markdown(""" **PENJELASAN KOLOM**
    1. **Pregnancies**
       * Deskripsi: Jumlah kehamilan yang pernah dialami oleh pasien wanita.
       * Statistik:
         - Nilai rata-rata (mean): 3.85
         - Rentang: 0 hingga 17
    2. **Glucose**
       * Deskripsi: Tingkat glukosa darah (mg/dL) yang menjadi indikator penting dalam diabetes.
       * Statistik:
         - Nilai rata-rata: 120.89
         - Nilai minimum: 0
         - Nilai maksimum: 199
         - Catatan: Nilai 0 mungkin mewakili data yang hilang atau tidak tercatat.
    3. **BloodPressure**
       * Deskripsi: Tekanan darah diastolik (mm Hg), menunjukkan tekanan darah di antara detak jantung.
       * Statistik:
         - Nilai rata-rata: 69.10
         - Rentang: 0 hingga 122
    4. **SkinThickness**
       * Deskripsi: Ketebalan lipatan kulit (mm), yang memberikan indikasi lemak tubuh.
       * Statistik:
         - Nilai rata-rata: 20.53
         - Nilai maksimum: 99
    5. **Insulin**
       * Deskripsi: Tingkat insulin serum (IU/mL), digunakan untuk memahami sensitivitas insulin tubuh.
       * Statistik:
         - Rata-rata: 79.80
         - Nilai maksimum: 846
    6. **BMI**
       * Deskripsi: Indeks Massa Tubuh (Body Mass Index), dihitung berdasarkan berat badan (kg) dan tinggi badan (m²).
       * Statistik:
         - Nilai rata-rata: 31.99
         - Kategori: Termasuk kategori berat badan berlebih.
    7. **DiabetesPedigreeFunction**
       * Deskripsi: Skor fungsi diabetes berdasarkan riwayat keluarga, menunjukkan risiko genetik.
       * Statistik:
         - Rata-rata: 0.47
         - Nilai tertinggi: 2.42
    8. **Age**
       * Deskripsi: Usia pasien (tahun).
       * Statistik:
         - Rata-rata usia: 33
         - Rentang: 21 hingga 81
    9. **Outcome**
       * Deskripsi: Hasil diagnostik diabetes pasien.
         - 0: Tidak terdiagnosis diabetes (Negatif).
         - 1: Terdiagnosis diabetes (Positif).
""")
    st.markdown(""" **PENJELASAN STATISTIK**
* **Count** : Jumlah data non-kosong untuk setiap kolom adalah 768, menunjukkan dataset lengkap tanpa nilai kosong.
* **Mean** : Rata-rata nilai dalam setiap kolom.
* **Std** : Standar deviasi, menunjukkan seberapa tersebar data dari nilai rata-rata.
* **Min/Max** : Nilai minimum dan maksimum untuk setiap kolom.
* **25%, 50%, 75%** : Kuartil 1, median, dan kuartil 3, memberikan distribusi data untuk masing-masing fitur.
""")
    
# Menu: Visualisasi Data
elif menu == "Visualisasi Data":
    st.title("Visualisasi Data")
    data = load_data()

    st.subheader("Distribusi Kadar Glukosa")
    fig, ax = plt.subplots()
    sns.histplot(data['Glucose'], kde=True, color="blue", ax=ax)
    st.pyplot(fig)
    st.markdown(""" Visualsasi data tersebut menampilkan distribusi kadar glukosa dalam bentuk histogram, yang menunjukkan frekuensi data kadar glukosa dalam rentang tertentu, dengan kurva distribusi normal (garis biru) untuk mempermudah interpretasi pola distribusi. """)
    st.markdown(""" Grafik distribusi kadar glukosa menunjukkan bahwa: """)
    st.markdown("""   
        * Sebagian besar individu memiliki kadar glukosa antara 75-150, dengan puncak sekitar 100-125, yang sering dikaitkan dengan kondisi normal atau prediabetes.
        * Ada kadar glukosa tinggi hingga 200, menunjukkan kemungkinan diabetes yang tidak terkontrol.
        * Nilai 0 pada glukosa kemungkinan data yang salah atau hilang, karena tidak realistis secara medis.
        * Distribusi cenderung normal dengan kemiringan ke kanan, menunjukkan beberapa individu memiliki kadar glukosa sangat tinggi. """)

    st.subheader("Perbandingan Outcome (Negatif vs Positif Diabetes)")
    fig, ax = plt.subplots()
    sns.countplot(x='Outcome', data=data, palette="Set2", ax=ax)
    ax.set_xticklabels(["Negatif Diabetes", "Positif Diabetes"])
    st.pyplot(fig)
    st.markdown(""" Grafik perbandingan outcome menunjukkan bahwa jumlah individu dengan diabetes negatif (tidak diabetes) jauh lebih tinggi dibandingkan dengan yang diabetes positif.Hal ini menunjukkan bahwa sebagian besar populasi dalam dataset cenderung tidak menderita diabetes, meskipun terdapat sejumlah individu yang teridentifikasi positif. Perbandingan ini menggambarkan prevalensi diabetes yang lebih rendah dibandingkan kondisi non-diabetes dalam kelompok data tersebut. """)

    st.subheader("Heatmap Korelasi Antar Fitur")
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = data.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    st.markdown(""" Heatmap menunjukkan hubungan antar fitur dalam dataset: """)
    st.markdown(""" 
                * Korelasi tinggi: Age dan Pregnancies (0.54), serta Glucose dan Outcome (0.47).
                * Korelasi rendah: Sebagian besar fitur, seperti SkinThickness dan Age (-0.11), menunjukkan hubungan lemah. """)

    st.subheader("WordCloud Fitur")
    wordcloud_data = ' '.join(data.columns)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(wordcloud_data)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)
    st.markdown(""" Wordcloud fitur menggambarkan frekuensi atau pentingnya fitur dalam dataset secara visual. Semakin besar ukuran teks suatu fitur, semakin sering fitur tersebut muncul atau semakin besar pengaruhnya dalam analisis atau model prediksi. Wordcloud ini membantu mengidentifikasi fitur utama secara cepat. """)

# Menu: Prediksi Diabetes
elif menu == "Prediksi Diabetes":
    st.title("Prediksi Diabetes")
    st.write("Masukkan data kesehatan Anda di bawah ini untuk memprediksi kemungkinan diabetes.")

    # Input pengguna
    pregnancies = st.number_input("Jumlah Kehamilan", min_value=0, max_value=20, value=4)
    glucose = st.number_input("Kadar Glukosa", min_value=0, max_value=200, value=85)
    blood_pressure = st.number_input("Tekanan Darah (mm Hg)", min_value=0, max_value=150, value=70)
    skin_thickness = st.number_input("Ketebalan Kulit (mm)", min_value=0, max_value=100, value=20)
    insulin = st.number_input("Insulin (IU/mL)", min_value=0, max_value=900, value=79)
    bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
    age = st.number_input("Usia", min_value=0, max_value=120, value=33)

    # Prediksi
    if st.button("Prediksi"):
        try:
            # Membuat DataFrame dari input pengguna
            input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]], 
                                      columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                                               'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
            
            # Load model
            model = load_model()
            
            # Prediksi
            prediction = model.predict(input_data)[0]
            result = "Positif Diabetes" if prediction == 1 else "Negatif Diabetes"
            
            st.success(f"Hasil prediksi: **{result}**")
        except Exception as e:
            st.error(f"Terjadi kesalahan dalam prediksi: {e}")

        st.markdown (""" Kesimpulan """)
        st.markdown (""" * **Positif Diabetes** : Lakukan tindakan pencegahan lebih serius dan segera konsultasikan dengan dokter. """)
        st.markdown (""" * **Negatif Diabetes** : Tetap waspada, jaga pola hidup sehat, dan lakukan pemeriksaan rutin jika memiliki faktor risiko. """)