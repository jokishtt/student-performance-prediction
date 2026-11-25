
import streamlit as st
import pandas as pd
import joblib

# Memuat model Machine Learning
model = joblib.load("student_performance_model.pkl")

# Konfigurasi halaman
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered"
)

# Judul aplikasi
st.title("🎓 Student Performance Prediction")
st.write(
    "Aplikasi prediksi nilai ujian siswa menggunakan "
    "Machine Learning dengan algoritma Linear Regression."
)

st.divider()

st.subheader("Masukkan Data Siswa")

# Form input
with st.form("prediction_form"):

    st.markdown("### 1. Data Akademik")

    col1, col2 = st.columns(2)

    with col1:
        hours_studied = st.number_input(
            "Jam Belajar",
            min_value=1,
            max_value=43,
            value=10
        )

        attendance = st.number_input(
            "Kehadiran (%)",
            min_value=60,
            max_value=100,
            value=85
        )

        previous_scores = st.number_input(
            "Nilai Sebelumnya",
            min_value=50,
            max_value=100,
            value=75
        )

    with col2:
        sleep_hours = st.number_input(
            "Jam Tidur",
            min_value=4,
            max_value=10,
            value=7
        )

        tutoring_sessions = st.number_input(
            "Jumlah Sesi Les",
            min_value=0,
            max_value=8,
            value=2
        )

        physical_activity = st.number_input(
            "Aktivitas Fisik",
            min_value=0,
            max_value=6,
            value=3
        )

    st.markdown("### 2. Lingkungan dan Kebiasaan Siswa")

    parental_involvement = st.selectbox(
        "Keterlibatan Orang Tua",
        ["High", "Medium", "Low"]
    )

    access_to_resources = st.selectbox(
        "Akses Sumber Belajar",
        ["High", "Medium", "Low"]
    )

    extracurricular_activities = st.selectbox(
        "Kegiatan Ekstrakurikuler",
        ["Yes", "No"]
    )

    motivation_level = st.selectbox(
        "Tingkat Motivasi",
        ["High", "Medium", "Low"]
    )

    internet_access = st.selectbox(
        "Akses Internet",
        ["Yes", "No"]
    )

    family_income = st.selectbox(
        "Pendapatan Keluarga",
        ["High", "Medium", "Low"]
    )

    teacher_quality = st.selectbox(
        "Kualitas Guru",
        ["High", "Medium", "Low"]
    )

    school_type = st.selectbox(
        "Jenis Sekolah",
        ["Public", "Private"]
    )

    peer_influence = st.selectbox(
        "Pengaruh Teman",
        ["Positive", "Neutral", "Negative"]
    )

    learning_disabilities = st.selectbox(
        "Kesulitan Belajar",
        ["No", "Yes"]
    )

    parental_education_level = st.selectbox(
        "Pendidikan Orang Tua",
        ["High School", "College", "Postgraduate"]
    )

    distance_from_home = st.selectbox(
        "Jarak Rumah ke Sekolah",
        ["Near", "Moderate", "Far"]
    )

    gender = st.selectbox(
        "Jenis Kelamin",
        ["Male", "Female"]
    )

    submit = st.form_submit_button(
        "Prediksi Nilai Ujian",
        use_container_width=True
    )

# Proses prediksi
if submit:

    # Menghitung fitur hasil Feature Engineering
    study_attendance_index = (
        hours_studied * attendance / 100
    )

    # Menyiapkan data sesuai kolom saat training
    input_data = pd.DataFrame([{
        "Hours_Studied": hours_studied,
        "Attendance": attendance,
        "Parental_Involvement": parental_involvement,
        "Access_to_Resources": access_to_resources,
        "Extracurricular_Activities": extracurricular_activities,
        "Sleep_Hours": sleep_hours,
        "Previous_Scores": previous_scores,
        "Motivation_Level": motivation_level,
        "Internet_Access": internet_access,
        "Tutoring_Sessions": tutoring_sessions,
        "Family_Income": family_income,
        "Teacher_Quality": teacher_quality,
        "School_Type": school_type,
        "Peer_Influence": peer_influence,
        "Physical_Activity": physical_activity,
        "Learning_Disabilities": learning_disabilities,
        "Parental_Education_Level": parental_education_level,
        "Distance_from_Home": distance_from_home,
        "Gender": gender,
        "Study_Attendance_Index": study_attendance_index
    }])

    # Melakukan prediksi
    prediction = model.predict(input_data)[0]

    st.divider()
    st.subheader("Hasil Prediksi")

    st.metric(
        label="Prediksi Nilai Ujian",
        value=f"{prediction:.2f}"
    )

    st.success(
        "Prediksi berhasil dilakukan menggunakan model Linear Regression."
    )

    st.caption(
        "Hasil ini merupakan estimasi berdasarkan pola data pelatihan, "
        "bukan nilai ujian yang dijamin."
    )
